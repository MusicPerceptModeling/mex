from functools import reduce
import math
from typing import List

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import wilcoxon
from datasets import Dataset, concatenate_datasets

from ..benchmark import PandasTestResult, Task
from ..metadata.segment import SegmentMetaData
from ..util.welford import merge as welford_merge, std as welford_std

import os
import matplotlib.pyplot as plt
import mne

from mtrf.model import TRF

# Ensure your local stats utils and plotting functions are imported at the top of the file:
# from eeg_utils.utils.utils_statistics import jackknife_ttest
# from .your_plot_file import plot_static_score_topomap, plot_participant_trf_topomaps, plot_group_level_topomap

class NeuralEncoding(Task):
    def __init__(
        self,
        tmin: float = -0.1,
        tmax: float = 0.7,
        use_banded: bool = False,
        n_perms: int = None
    ):
        # Register the expected metadata schema
        super().__init__(EEGMetaData)
        self.tmin = tmin
        self.tmax = tmax
        self.use_banded = use_banded
        self.n_perms = n_perms

    def _run_task(self, midm: Dataset, metadata: Dataset, **kwargs) -> EEGTRFResult:
        """
        midm: Contains 'id', 'envelope', 'ic' (the model outputs)
        metadata: Contains 'id', 'subject', 'eeg_data', 'ch_names', 'fs'
        """

        # 1. Merge the model outputs with the EEG metadata based on unique 'id'
        midm_df = midm.to_pandas()
        meta_df = metadata.to_pandas()
        merged_df = pd.merge(midm_df, meta_df, on="id")

        if merged_df.empty:
            raise ValueError("Merging midm and metadata resulted in an empty dataframe. Check 'id' alignment.")

        results_records = []
        weights_dictionary = {}

        fs = merged_df.iloc[0]['fs']
        ch_names = merged_df.iloc[0]['ch_names']

        # 2. Iterate over subjects to fit TRFs
        for subject, group in merged_df.groupby('subject'):
            stimuli_list = []
            eeg_list = []

            # Format lists for mTRFpy
            for _, row in group.iterrows():
                # Assumes 'envelope' and 'ic' are present in the MIDM dataset
                stim = np.column_stack((row['envelope'], row['ic']))
                eeg = np.array(row['eeg_data'])
                stimuli_list.append(stim)
                eeg_list.append(eeg)

            # 3. Fit Cross-Validated Models (Your core mathematical logic)
            z_full, z_base, delta_z, opt_lambda = compare_direct_models(
                stimuli_list, eeg_list, fs, use_banded=self.use_banded
            )

            delta_z = np.nan_to_num(delta_z, nan=0.0)
            z_base = np.nan_to_num(z_base, nan=0.0)

            # Store channel-level correlation metrics
            for ch_idx, ch_name in enumerate(ch_names):
                results_records.append({
                    "subject": subject,
                    "channel": ch_name,
                    "z_base": z_base[ch_idx],
                    "delta_z": delta_z[ch_idx]
                })

            # 4. Fit Final Model to Extract Weights
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

            # Note: Permutation logic can be implemented here exactly as you did
            # in your script, appending the null distributions to results_records
            # if self.n_perms is True.

        metrics_df = pd.DataFrame.from_records(results_records)

        # 5. Return the custom Data Object
        return EEGTRFResult(
            correlation_metrics=metrics_df,
            weights_dict=weights_dictionary,
            ch_names=ch_names,
            fs=fs,
            tmin=self.tmin
        )



from ..benchmark import Result

class NeuralEncodingResult(Result):
    """
    Holds exactly the mathematical artifacts needed: correlation matrices and TRF weights.
    Handles optional plotting on save.
    """
    def __init__(
        self,
        correlation_metrics: pd.DataFrame,
        weights_dict: dict,
        ch_names: list,
        fs: float,
        tmin: float
    ):
        self.correlation_metrics = correlation_metrics
        self.weights_dict = weights_dict
        self.ch_names = list(ch_names)
        self.fs = fs
        self.tmin = tmin

    def __str__(self) -> str:
        return self.correlation_metrics.to_string(index=False)

    def save(self, path: str, plot_graphs: bool = False):
        """
        Persists the matrices to disk. Triggers plotting only if requested.
        """
        os.makedirs(path, exist_ok=True)

        #saving correlation matrices (dataframe to parquet)
        metrics_path = os.path.join(path, "trf_correlation_metrics.parquet")
        self.correlation_metrics.to_parquet(metrics_path)

        #saving TRF weights (dictionary to npz)
        weights_path = os.path.join(path, "trf_weights.npz")
        np.savez(weights_path, **self.weights_dict)

        print(f"Artifacts saved to {path}")

        #optional topomaps and time delay weight graphs
        if plot_graphs:
            print("Plotting graphs as requested...")
            self._generate_plots(path)

    @classmethod
    def load(cls, path: str) -> 'EEGTRFResult':
        """Reconstructs the Result object from a saved directory."""
        metrics_df = pd.read_parquet(os.path.join(path, "trf_correlation_metrics.parquet"))

        weights_npz = np.load(os.path.join(path, "trf_weights.npz"))
        weights_dict = {key: weights_npz[key] for key in weights_npz.files}

        #extracting ch_names dynamically
        ch_names = metrics_df['channel'].unique().tolist()

        #standard values for reconstruction (could be extended to save in a JSON)
        return cls(metrics_df, weights_dict, ch_names=ch_names, fs=64.0, tmin=-0.1)

    ###########################
    #Plotting topomap functions
    ###########################
    @staticmethod
    def plot_static_score_topomap(
        scores: np.ndarray,
        ch_names: list,
        subj_id: str,
        score_type: str = "Δz (Predictive Enhancement)",
        cmap: str = "Reds",
        vmin: float = None,
        vmax: float = None
    ):
        """
        Plots a static scalp topomap for a 1D array of channel scores (e.g., Pearson's r or Δz).

        Parameters:
        - scores: 1D numpy array of length n_channels.
        - ch_names: List of strings matching the standard 10-20 system. TO DO : get the true montage from the data!
        - subj_id: String identifier for the plot title.
        - score_type: The metric being plotted (used for the colorbar label).
        - cmap: Matplotlib colormap (e.g., 'Reds' for positive enhancements, 'RdBu_r' if including negative).
        - vmin, vmax: Optional fixed limits for the color scale (useful for comparing subjects).
        """

        # ensure scores is a flat 1D array
        scores = np.asarray(scores).flatten()
        if len(scores) != len(ch_names):
            raise ValueError(f"Length of scores ({len(scores)}) does not match length of ch_names ({len(ch_names)}).")

        # creating MNE Info object
        # We use a dummy sampling rate (1000) because it is irrelevant for static spatial plots
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')

        # apply standard 10-05 scalp montage TO DO : get the montage from the data itself !
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, match_case=False, on_missing='warn')
        except ValueError as e:
            print(f"Warning: Montage mapping issue. {e}")

        fig, ax = plt.subplots(figsize=(6, 5))

        # topomap using the 1D scores directly
        im, _ = mne.viz.plot_topomap(
            scores,
            info,
            axes=ax,
            cmap=cmap,
            vlim=(vmin, vmax),
            show=False,
            extrapolate='head' #prevents bleeding outside the head circle
        )

        ax.set_title(f"Spatial Distribution of Model Accuracy\nParticipant: {subj_id}")

        #colorbar to interpret the values
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(score_type)

        fig.tight_layout()
        return fig


    @staticmethod
    def plot_participant_trf_topomaps(
        trf_model: TRF,
        ch_names: list,
        fs: float,
        tmin: float,
        subj_id: str,
        feature_idx: int = 1,
        feature_name: str = "IC",
        times_to_plot: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ):
        # extracting weights and transpose for MNE
        weights = trf_model.weights[feature_idx, :, :]
        weights_mne = weights.T

        # creating MNE Info and montage
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, match_case=False, on_missing='ignore')
        except ValueError:
            pass

        evoked = mne.EvokedArray(weights_mne, info, tmin=tmin)

        # calculating a robust maximum limit using the 95th percentile.
        # this prevents the extreme spike at 0.000s from washing out the other maps.
        robust_vmax = np.percentile(np.abs(weights_mne), 95)

        fig = evoked.plot_topomap(
            times=times_to_plot,
            ch_type='eeg',
            time_unit='s',
            colorbar=True,
            show=False,
            scalings=dict(eeg=1),             # Stop MNE from multiplying by 1,000,000
            units=dict(eeg='TRF Weight'),     # Renaming the colorbar label
            vlim=(-robust_vmax, robust_vmax)
        )

        fig.suptitle(f"TRF Spatial Weights over Time\nParticipant: {subj_id} | Predictor: {feature_name}", fontsize=14)

        return fig

    @staticmethod
    def plot_group_level_topomap(
        mean_scores: np.ndarray,
        ch_names: list,
        significant_mask: np.ndarray = None,
        score_type: str = "Mean Δz (Predictive Enhancement)",
        cmap: str = "Reds",
        vmin: float = None,
        vmax: float = None
    ):
        mean_scores = np.asarray(mean_scores).flatten()
        if len(mean_scores) != len(ch_names):
            raise ValueError("Length of mean_scores must match length of ch_names.")

        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, match_case=False, on_missing='ignore') # Set to ignore to stop spam
        except ValueError:
            pass

        mask_params = dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=0, markersize=6)

        fig, ax = plt.subplots(figsize=(6, 5))

        im, _ = mne.viz.plot_topomap(
            data=mean_scores,
            pos=info,
            axes=ax,
            cmap=cmap,
            vlim=(vmin, vmax),
            mask=significant_mask,
            mask_params=mask_params,
            show=False,
            extrapolate='head'
        )

        ax.set_title("Group-Level Predictive Enhancement\n(White dots = FDR corrected p < 0.05)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(score_type)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_behavioral_trf_weights(trf_model: TRF, fs: float, tmin: float, tmax: float, subj_id: str):
        """
        Plots a 1D line graph of the TRF weights over time for behavioral data.
        Specifically isolates Information Content (IC) predicting Arousal.
        """
        # Feature 1 = Information Content (0 is Envelope)
        # Target 1 = Arousal (0 is Valence)
        arousal_ic_weights = trf_model.weights[1, :, 1]

        # matching X-axis based on the actual weight array length
        n_lags = arousal_ic_weights.shape[0]
        lags = np.linspace(tmin, tmax, n_lags)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(lags, arousal_ic_weights, color='firebrick', linewidth=2, label="IC -> Arousal")
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)

        ax.set_title(f"Temporal TRF Weights: Surprisal driving Arousal\nParticipant: {subj_id}")
        ax.set_xlabel("Time Lag (seconds)")
        ax.set_ylabel("TRF Weight (Influence on Joystick)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig


    @staticmethod
    def plot_group_behavioral_weights(all_weights: list, fs: float, tmin: float, tmax: float):
        """
        Plots the average 1D temporal TRF weights across all participants,
        including a shaded region for the Standard Error of the Mean (SEM).
        """
        weights_matrix = np.stack(all_weights, axis=0)

        mean_weights = np.mean(weights_matrix, axis=0)
        sem_weights = np.std(weights_matrix, axis=0) / np.sqrt(weights_matrix.shape[0])

        n_lags = mean_weights.shape[0]
        lags = np.linspace(tmin, tmax, n_lags)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(lags, mean_weights, color='firebrick', linewidth=2, label="Mean IC -> Arousal")

        ax.fill_between(
            lags,
            mean_weights - sem_weights,
            mean_weights + sem_weights,
            color='firebrick',
            alpha=0.3,
            label="Standard Error"
        )

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)

        ax.set_title("Group-Level Temporal TRF: Surprisal driving Arousal")
        ax.set_xlabel("Time Lag (seconds)")
        ax.set_ylabel("Mean TRF Weight (Influence on Joystick)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig


    ################################################
    #Preprocessing and eeg preparation and alignment
    ################################################
    #quality evaluation :
    @staticmethod
    def plot_ica_diagnostics(raw_before, ica, subj_id, outdir="results"):
        raw_eeg_only = raw_before.copy().pick(picks='eeg')

        diag_dir = f"{outdir}/diagnostics/{subj_id}"
        os.makedirs(diag_dir, exist_ok=True)

        # 1. Plot ICA Sources (Time courses)
        fig_sources = ica.plot_sources(raw_eeg_only, show=False)
        fig_sources.savefig(f"{diag_dir}/ica_sources.png")

        # 2. Plot component properties (only for excluded ones)
        if ica.exclude:
            # passing the EEG-only object here
            figs = ica.plot_properties(raw_eeg_only, picks=ica.exclude, show=False)
            for i, fig in enumerate(figs):
                fig.savefig(f"{diag_dir}/excluded_prop_{ica.exclude[i]}.png")
                plt.close(fig)

        #ICA component plots (topomaps)
        fig_topo = ica.plot_components(show=False)
        fig_topo.savefig(f"{diag_dir}/ica_components.png")

        plt.close('all')

    def _generate_plots(self, outdir: str):
        """Internal method routing data to your existing Matplotlib/MNE functions."""
        plots_dir = os.path.join(outdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        print("  -> Generating Subject-Level Topomaps...")

        # We need a predictable order for channels to ensure numpy arrays align
        ch_order = self.ch_names

        # 1. Subject-level plots
        for subject, group in self.correlation_metrics.groupby("subject"):
            # Ensure the scores are ordered exactly as self.ch_names
            group = group.set_index('channel').reindex(ch_order)
            scalp_scores = group['delta_z'].fillna(0.0).values

            # --- Plot A: Subject Delta Z Topomap ---
            fig_dz = plot_static_score_topomap(
                scores=scalp_scores,
                ch_names=ch_order,
                subj_id=subject,
                score_type="Δz (Full - Baseline)"
            )
            fig_dz.savefig(f"{plots_dir}/topomap_{subject}_delta_z.png", dpi=300)
            plt.close(fig_dz)

            # --- Plot B: Subject TRF Weights Topomap ---
            if subject in self.weights_dict:
                # Reconstruct a dummy TRF object to satisfy the plotting function's signature
                dummy_trf = TRF(direction=1)
                dummy_trf.weights = self.weights_dict[subject]

                fig_weights = plot_participant_trf_topomaps(
                    trf_model=dummy_trf,
                    ch_names=ch_order,
                    fs=self.fs,
                    tmin=self.tmin,
                    subj_id=subject,
                    feature_idx=1,
                    feature_name="Information Content (IC)",
                    times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                )
                fig_weights.savefig(f"{plots_dir}/topomap_song_{subject}_IC_weights.png", dpi=300)
                plt.close(fig_weights)

        # 2. Group-level plots
        print("  -> Computing Group-Level Statistics for Final Topomap...")

        # Pivot the dataframe to get a matrix of shape (n_subjects, n_channels)
        pivot_df = self.correlation_metrics.pivot(index='subject', columns='channel', values='delta_z')
        pivot_df = pivot_df[ch_order] # Enforce channel order
        group_delta_z = pivot_df.values

        mean_delta_z = np.mean(group_delta_z, axis=0)

        # Compute the significance mask using Jackknife/Wilcoxon
        p_values = []
        for ch in range(group_delta_z.shape[1]):
            try:
                # Requires: from eeg_utils.utils.utils_statistics import jackknife_ttest
                _, p = jackknife_ttest(group_delta_z[:, ch], alternative='greater')
            except Exception:
                res = wilcoxon(group_delta_z[:, ch], alternative='greater')
                p = res.pvalue
            p_values.append(p)

        p_values = np.array(p_values)
        reject_null, p_values_fdr = mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')

        # --- Plot C: Group Mean Delta Z Topomap ---
        fig_group = plot_group_level_topomap(
            mean_scores=mean_delta_z,
            ch_names=ch_order,
            significant_mask=reject_null,
            score_type="Mean Δz",
            cmap="Reds",
            vmin=0.0
        )
        fig_group.savefig(f"{plots_dir}/group_level_enhancement_song.png", dpi=300)
        plt.close(fig_group)
        print(f"All plots successfully saved to {plots_dir}/")
