from functools import reduce
import math
from typing import List

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import t, ttest_1samp, wilcoxon, shapiro, zscore
from datasets import Dataset, concatenate_datasets

from ..benchmark import Task, Result
from ..metadata.segment import SegmentMetaData
from ..util.welford import merge as welford_merge, std as welford_std

import os
import matplotlib.pyplot as plt
import mne

from mtrf.model import TRF
from mtrf.stats import nested_crossval, crossval
from ..metadata.neural import NeuralEncodingMetaData


class NeuralEncodingTask(Task):
    def __init__(
        self,
        tmin: float = -0.1,
        tmax: float = 0.7,
        use_banded: bool = False,
        n_perms: int = None,
        midm_column: str = "ic",
        env_dt: float = 0.015625,  # 64Hz envelope
        ic_dt: float = 0.0928798   # pa ARDM framerate
    ):
        super().__init__(NeuralEncodingMetaData)
        self.tmin = tmin
        self.tmax = tmax
        self.use_banded = use_banded
        self.n_perms = n_perms
        self.midm_column = midm_column
        self.env_dt = env_dt
        self.ic_dt = ic_dt

    @staticmethod
    def fisher_z_transform(r: np.ndarray) -> np.ndarray:
        """applies Fisher z-transformation to Pearson's r values"""
        r_clipped = np.clip(r, -0.999, 0.999)
        return np.arctanh(r_clipped)

    @staticmethod
    def compare_direct_models(stimuli_list, eeg_list, fs, use_banded=False):
        tmin, tmax = -0.1, 0.7
        k_folds = -1
        regularization = np.logspace(-1, 6, 8)
        trf = TRF(direction=1)

        # --- Baseline model (Energy only) ---
        baseline_stimuli = [stim[:, 0:1] for stim in stimuli_list]
        print("Fitting Baseline Model (Energy only)...")
        r_base, best_lambdas_base = nested_crossval(
            trf, baseline_stimuli, eeg_list, fs, tmin, tmax, regularization,
            k=k_folds, average=True, verbose=False
        )
        opt_lambda_env = np.median(best_lambdas_base)

        r_base = crossval(
            trf, baseline_stimuli, eeg_list, fs, tmin, tmax, opt_lambda_env,
            k=k_folds, average=False, verbose=False
        )
        if r_base.ndim == 2:
            r_base = np.mean(r_base, axis=0)
        z_base = NeuralEncodingTask.fisher_z_transform(r_base)
        print(f'Best energy-only lambda: {opt_lambda_env}')

        # --- Full model (Energy + IC) ---
        if use_banded:
            print("Fitting Full Model with banded Regularization (manual list grid search)...")
            best_r = -np.inf
            opt_lambda_full = [1.0, 1.0]
            n_epochs = len(stimuli_list)

            reg_env = np.logspace(-1, 3, 10)
            reg_ic = np.logspace(-1, 3, 10)

            for l_env in reg_env:
                for l_ic in reg_ic:
                    fold_rs = []
                    for i in range(n_epochs):
                        val_stim = stimuli_list[i]
                        val_eeg = eeg_list[i]

                        train_stim = [stimuli_list[j] for j in range(n_epochs) if j != i]
                        train_eeg = [eeg_list[j] for j in range(n_epochs) if j != i]

                        trf_temp = TRF(direction=1)

                        trf_temp.train(
                            train_stim, train_eeg, fs, tmin, tmax,
                            regularization=[l_env, l_ic], bands=[1, 1]
                        )

                        _, r = trf_temp.predict(val_stim, val_eeg)
                        fold_rs.append(np.mean(r))

                    mean_grid_r = np.mean(fold_rs)
                    if mean_grid_r > best_r:
                        best_r = mean_grid_r
                        opt_lambda_full = [l_env, l_ic]

            print(f'Best banded lambdas [Env, IC]: {opt_lambda_full}')

            final_rs = []
            for i in range(n_epochs):
                val_stim = stimuli_list[i]
                val_eeg = eeg_list[i]

                train_stim = [stimuli_list[j] for j in range(n_epochs) if j != i]
                train_eeg = [eeg_list[j] for j in range(n_epochs) if j != i]

                trf_final = TRF(direction=1)
                trf_final.train(
                    train_stim, train_eeg, fs, tmin, tmax,
                    regularization=opt_lambda_full, bands=[1, 1]
                )
                _, r = trf_final.predict(val_stim, val_eeg)
                final_rs.append(r)

            r_full = np.mean(final_rs, axis=0)

        else:
            r_full, best_lambdas_full = nested_crossval(
                trf, stimuli_list, eeg_list, fs, tmin, tmax, regularization,
                k=k_folds, average=True, verbose=False
            )
            opt_lambda_full = np.median(best_lambdas_full)
            r_full = crossval(
                trf, stimuli_list, eeg_list, fs, tmin, tmax, opt_lambda_full,
                k=k_folds, average=False, verbose=False
            )
            if r_full.ndim == 2:
                r_full = np.mean(r_full, axis=0)

        z_full = NeuralEncodingTask.fisher_z_transform(r_full)
        delta_z = z_full - z_base

        return z_full, z_base, delta_z, opt_lambda_full

    def _run_task(self, midm: Dataset, metadata: Dataset, **kwargs) -> 'NeuralEncodingResult':
        midm_df = midm.to_pandas()
        meta_df = metadata.to_pandas()
        
        #cleaning IDs to simple integers 
        midm_df["id"] = midm_df["id"].astype(int)
        meta_df["stimulus_id"] = meta_df["stimulus_id"].str.replace("trial_", "").astype(int)

        merged_df = pd.merge(midm_df, meta_df, left_on="id", right_on="stimulus_id")
        
        if merged_df.empty:
            raise ValueError("Merging midm and metadata resulted in an empty dataframe. Check 'id' alignment.")

        results_records = []
        weights_dictionary = {}
        ch_names = merged_df.iloc[0]['ch_names']

        for subject, group in merged_df.groupby('subject'):
            stimuli_list = []
            eeg_list = []
            
            #default fallback, but to be dynamically overwritten per row
            fs = 64.0 

            for _, row in group.iterrows():
                #reading the true sampling rate to prevent time-warping
                fs = float(row['fs'])
                
                env = np.array(list(row['envelope']), dtype=float)
                feat = np.array(list(row[self.midm_column]), dtype=float)

                eeg = np.vstack(row['eeg_data']).astype(float)
                if eeg.ndim == 2 and eeg.shape[0] < eeg.shape[1]:
                    eeg = eeg.T
                elif eeg.ndim == 1:
                    eeg = eeg[:, np.newaxis]

                eeg_length = eeg.shape[0]
                eeg_time = np.arange(eeg_length) / fs
                env_time = np.arange(len(env)) * self.env_dt

                if 'time' in row and row['time'] is not None:
                    ic_time = np.array(list(row['time']), dtype=float)
                elif 'ic_time' in row and row['ic_time'] is not None:
                    ic_time = np.array(list(row['ic_time']), dtype=float)
                else:
                    ic_time = np.arange(len(feat)) * self.ic_dt

                from scipy.interpolate import interp1d
                ic_interpolator = interp1d(ic_time, feat, kind='linear', bounds_error=False, fill_value=0.0)
                ic_aligned = ic_interpolator(eeg_time)

                env_interpolator = interp1d(env_time, env, kind='linear', bounds_error=False, fill_value=0.0)
                env_aligned = env_interpolator(eeg_time)

                stim = np.column_stack((env_aligned, ic_aligned))
                clean_stim = np.nan_to_num(zscore(stim, axis=0))
                clean_eeg = np.nan_to_num(zscore(eeg, axis=0))

                stimuli_list.append(clean_stim)
                eeg_list.append(clean_eeg)

            if not stimuli_list:
                continue
            
            #diagnostic print to confirm data integrity 
            print(f"[{subject}] Merged {len(stimuli_list)} valid epochs at {fs} Hz.")

            z_full, z_base, delta_z, opt_lambda = self.compare_direct_models(
                stimuli_list, eeg_list, fs, use_banded=self.use_banded
            )

            delta_z = np.nan_to_num(delta_z, nan=0.0)
            z_base = np.nan_to_num(z_base, nan=0.0)

            for ch_idx, ch_name in enumerate(ch_names):
                results_records.append({
                    "subject": subject,
                    "channel": ch_name,
                    "z_base": z_base[ch_idx],
                    "delta_z": delta_z[ch_idx]
                })

            final_trf = TRF(direction=1)
            if isinstance(opt_lambda, list):
                final_trf.train(
                    stimuli_list, eeg_list, fs, tmin=self.tmin, tmax=self.tmax,
                    regularization=opt_lambda, bands=[1, 1]
                )
            else:
                final_trf.train(
                    stimuli_list, eeg_list, fs, tmin=self.tmin, tmax=self.tmax,
                    regularization=opt_lambda
                )

            weights_dictionary[subject] = final_trf.weights

        metrics_df = pd.DataFrame.from_records(results_records)

        return NeuralEncodingResult(
            correlation_metrics=metrics_df,
            weights_dict=weights_dictionary,
            ch_names=ch_names,
            fs=fs,
            tmin=self.tmin,
            mid_feature_name=self.midm_column.upper()
        )


class NeuralEncodingResult(Result):
    def __init__(
        self,
        correlation_metrics: pd.DataFrame,
        weights_dict: dict,
        ch_names: list,
        fs: float,
        tmin: float,
        mid_feature_name: str = "MID"
    ):
        self.correlation_metrics = correlation_metrics
        self.weights_dict = weights_dict
        self.ch_names = list(ch_names)
        self.fs = fs
        self.tmin = tmin
        self.mid_feature_name = mid_feature_name

    def __str__(self) -> str:
        return self.correlation_metrics.to_string(index=False)

    def save(self, path: str, plot_graphs: bool = True, print_summary: bool = True):
        os.makedirs(path, exist_ok=True)
        metrics_path = os.path.join(path, "trf_correlation_metrics.parquet")
        self.correlation_metrics.to_parquet(metrics_path)
        weights_path = os.path.join(path, "trf_weights.npz")
        np.savez(weights_path, **self.weights_dict)
        print(f"Artifacts saved to {path}")

        if print_summary:
            self.print_summary()
        if plot_graphs:
            print("Plotting graphs as requested...")
            self._generate_plots(path)

    @classmethod
    def load(cls, path: str) -> 'NeuralEncodingResult':
        metrics_df = pd.read_parquet(os.path.join(path, "trf_correlation_metrics.parquet"))
        weights_npz = np.load(os.path.join(path, "trf_weights.npz"))
        weights_dict = {key: weights_npz[key] for key in weights_npz.files}
        ch_names = metrics_df['channel'].unique().tolist()
        return cls(metrics_df, weights_dict, ch_names=ch_names, fs=64.0, tmin=-0.1)
    
    def print_summary(self):
        print("\n==========================================")
        print(" Computing Group-Level Statistics...")
        print("==========================================")

        # Standard 10-05 scalp electrodes 
        eeg_only_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
            'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
        ]
        
        # Keep only the valid EEG names that actually exist in the dataset
        valid_eeg_names = [ch for ch in eeg_only_names if ch in self.ch_names]

        # Pivot matrices strictly aligned to the 30 EEG channels before stats
        pivot_base = self.correlation_metrics.pivot(index='subject', columns='channel', values='z_base')[valid_eeg_names]
        group_z_base = pivot_base.values
        mean_z_base = np.mean(group_z_base, axis=0)

        pivot_delta = self.correlation_metrics.pivot(index='subject', columns='channel', values='delta_z')[valid_eeg_names]
        group_delta_z = pivot_delta.values
        mean_delta_z = np.mean(group_delta_z, axis=0)

        p_values_base = []
        for ch in range(group_z_base.shape[1]):
            try:
                _, p = self.jackknife_ttest(group_z_base[:, ch], alternative='greater')
            except Exception:
                res = wilcoxon(group_z_base[:, ch], alternative='greater')
                p = res.pvalue
            p_values_base.append(p)

        p_values_base = np.array(p_values_base)
        reject_null_base, p_values_fdr_base = mne.stats.fdr_correction(p_values_base, alpha=0.05, method='indep')
        sig_ch_base = np.where(reject_null_base)[0]

        print(f"Overall Mean Scalp Envelope Prediction (z_base): {np.mean(mean_z_base):.4f}")
        if len(sig_ch_base) > 0:
            print("\nSignificant Envelope Channels (FDR Corrected):")
            for ch_idx in sig_ch_base:
                print(f" - {valid_eeg_names[ch_idx]}: z_base = {mean_z_base[ch_idx]:.4f} (p = {p_values_fdr_base[ch_idx]:.4f})")

        p_values_delta = []
        for ch in range(group_delta_z.shape[1]):
            try:
                _, p = self.jackknife_ttest(group_delta_z[:, ch], alternative='greater')
            except Exception:
                res = wilcoxon(group_delta_z[:, ch], alternative='greater')
                p = res.pvalue
            p_values_delta.append(p)

        # FDR correction strictly on the 30 EEG channels
        reject_null_delta, p_values_fdr_delta = mne.stats.fdr_correction(p_values_delta, alpha=0.05, method='indep')
        sig_ch_delta = np.where(reject_null_delta)[0]

        print(f"\nTotal Subjects Processed: {group_delta_z.shape[0]}")
        print(f"Overall Mean Scalp Enhancement: {np.mean(mean_delta_z):.4f}")
        print(f"Number of EEG channels with significant IC enhancement: {len(sig_ch_delta)} / {len(valid_eeg_names)}")

        if len(sig_ch_delta) > 0:
            print("\nSignificant Channels (FDR Corrected):")
            for ch_idx in sig_ch_delta:
                print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr_delta[ch_idx]:.4f})")

        print("\n==========================================")
        print(" Computing ROI-Averaged Statistics...")
        print("==========================================")
        roi_names = ['Fz', 'Cz', 'FC1', 'FC2', 'C3', 'C4', 'FC5', 'FC6', 'T7', 'T8', 'CP5', 'CP6']
        roi_indices = [valid_eeg_names.index(ch) for ch in roi_names if ch in valid_eeg_names]

        if roi_indices:
            roi_delta_z = group_delta_z[:, roi_indices]
            subj_roi_averages = np.mean(roi_delta_z, axis=1)
            mean_roi_enhancement = np.mean(subj_roi_averages)

            print(f"ROI Channels Used: {[valid_eeg_names[i] for i in roi_indices]}")
            print(f"Raw Mean ROI Enhancement: {mean_roi_enhancement:.4f}")

            p_val_roi, test_name_roi = self.test_roi_significance(subj_roi_averages)
            sig_str = "*** SIGNIFICANT ***" if p_val_roi < 0.05 else "(Not Significant)"
            print(f"ROI {test_name_roi}: p = {p_val_roi:.4f} {sig_str}")

        print("\n==========================================")
        print(" Computing Global-Averaged Statistics...")
        print("==========================================")
        subj_global_averages = np.mean(group_delta_z, axis=1)
        mean_global_enhancement = np.mean(subj_global_averages)
        print(f"Global Channels Used: All {len(valid_eeg_names)} electrodes")
        print(f"Raw Mean Global Enhancement: {mean_global_enhancement:.4f}")
        
        p_val_glob, test_name_glob = self.test_roi_significance(subj_global_averages)
        sig_str_glob = "*** SIGNIFICANT ***" if p_val_glob < 0.05 else "(Not Significant)"
        print(f"Global {test_name_glob}: p = {p_val_glob:.4f} {sig_str_glob}\n")

    @staticmethod
    def jackknife_ttest(data, alternative='greater'):
        n = len(data)
        if n <= 1:
            raise ValueError("Data must contain more than one sample for jackknife.")
        mean_obs = np.mean(data)
        jack_means = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jack_means[i] = np.mean(jack_sample)
        mean_jack = np.mean(jack_means)
        se_jack = np.sqrt(((n - 1) / n) * np.sum((jack_means - mean_jack)**2))
        if se_jack == 0:
            se_jack = 1e-10
        t_stat = mean_obs / se_jack
        df = n - 1
        if alternative == 'greater':
            p_val = t.sf(t_stat, df)
        elif alternative == 'less':
            p_val = t.cdf(t_stat, df)
        else:
            p_val = 2 * t.sf(np.abs(t_stat), df)
        return t_stat, p_val

    @staticmethod
    def test_roi_significance(roi_averages):
        stat_shap, p_shapiro = shapiro(roi_averages)
        if p_shapiro < 0.05:
            res = wilcoxon(roi_averages, alternative='greater')
            return res.pvalue, "Wilcoxon Signed-Rank"
        else:
            res = ttest_1samp(roi_averages, popmean=0.0, alternative='greater')
            return res.pvalue, "1-Sample T-Test"

    @staticmethod
    def plot_static_score_topomap(scores, ch_names, subj_id, score_type="Δz (Predictive Enhancement)", cmap="Reds", vmin=None, vmax=None):
        scores = np.asarray(scores).flatten()
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, match_case=False, on_missing='warn')
        except ValueError:
            pass

        fig, ax = plt.subplots(figsize=(6, 5))
        im, _ = mne.viz.plot_topomap(scores, info, axes=ax, cmap=cmap, vlim=(vmin, vmax), show=False, extrapolate='head')
        ax.set_title(f"Spatial Distribution of Model Accuracy\nParticipant: {subj_id}")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(score_type)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_participant_trf_topomaps(trf_model, ch_names, fs, tmin, subj_id, feature_idx=1, feature_name="IC", times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        weights_mne = trf_model.weights[feature_idx, :, :].T
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, match_case=False, on_missing='ignore')
        except ValueError:
            pass

        evoked = mne.EvokedArray(weights_mne, info, tmin=tmin)
        robust_vmax = np.percentile(np.abs(weights_mne), 95)
        fig = evoked.plot_topomap(
            times=times_to_plot, ch_type='eeg', time_unit='s', colorbar=True, show=False,
            scalings=dict(eeg=1), units=dict(eeg='TRF Weight'), vlim=(-robust_vmax, robust_vmax)
        )
        fig.suptitle(f"TRF Spatial Weights over Time\nParticipant: {subj_id} | Predictor: {feature_name}", fontsize=14)
        return fig

    @staticmethod
    def plot_group_level_topomap(mean_scores, ch_names, significant_mask=None, score_type="Mean Δz (Predictive Enhancement)", cmap="Reds", vmin=None, vmax=None):
        mean_scores = np.asarray(mean_scores).flatten()
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, match_case=False, on_missing='ignore')
        except ValueError:
            pass

        mask_params = dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=0, markersize=6)
        fig, ax = plt.subplots(figsize=(6, 5))
        im, _ = mne.viz.plot_topomap(
            data=mean_scores, pos=info, axes=ax, cmap=cmap, vlim=(vmin, vmax),
            mask=significant_mask, mask_params=mask_params, show=False, extrapolate='head'
        )
        ax.set_title("Group-Level Predictive Enhancement\n(White dots = FDR corrected p < 0.05)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(score_type)
        fig.tight_layout()
        return fig

    def _generate_plots(self, outdir: str):
        plots_dir = os.path.join(outdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        print("  -> Generating Subject-Level Topomaps...")
        
        eeg_only_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
            'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
        ]
        valid_eeg_names = [ch for ch in eeg_only_names if ch in self.ch_names]
        
        for subject, group in self.correlation_metrics.groupby("subject"):
            group = group.set_index('channel').reindex(valid_eeg_names)
            scalp_scores = group['delta_z'].fillna(0.0).values
            
            fig_dz = self.plot_static_score_topomap(scalp_scores, valid_eeg_names, subject, "Δz (Full - Baseline)")
            fig_dz.savefig(f"{plots_dir}/topomap_{subject}_delta_z.png", dpi=300)
            plt.close(fig_dz)

            if subject in self.weights_dict:
                dummy_trf = TRF(direction=1)
                dummy_trf.weights = self.weights_dict[subject]
                
                # Slice weights down to valid EEG channels for plotting
                eeg_indices = [self.ch_names.index(ch) for ch in valid_eeg_names]
                sliced_weights = dummy_trf.weights[:, :, eeg_indices]
                dummy_trf.weights = sliced_weights
                
                fig_weights = self.plot_participant_trf_topomaps(
                dummy_trf, valid_eeg_names, self.fs, self.tmin, subject, 1, 
                feature_name=self.mid_feature_name,
                times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                )
                fig_weights.savefig(f"{plots_dir}/topomap_song_{subject}_{self.mid_feature_name}_weights.png", dpi=300)
                plt.close(fig_weights)

        print("  -> Computing Group-Level Statistics for Final Topomap...")
        pivot_df = self.correlation_metrics.pivot(index='subject', columns='channel', values='delta_z')[valid_eeg_names]
        group_delta_z = pivot_df.values
        mean_delta_z = np.mean(group_delta_z, axis=0)

        p_values = []
        for ch in range(group_delta_z.shape[1]):
            try:
                _, p = self.jackknife_ttest(group_delta_z[:, ch], alternative='greater')
            except Exception:
                res = wilcoxon(group_delta_z[:, ch], alternative='greater')
                p = res.pvalue
            p_values.append(p)

        p_values = np.array(p_values)
        reject_null, _ = mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')

        fig_group = self.plot_group_level_topomap(mean_delta_z, valid_eeg_names, reject_null, "Mean Δz", "Reds", 0.0)
        fig_group.savefig(f"{plots_dir}/group_level_enhancement_song.png", dpi=300)
        plt.close(fig_group)
        print(f"All plots successfully saved to {plots_dir}/")
