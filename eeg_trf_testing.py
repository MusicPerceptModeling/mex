import numpy as np
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval

#from eeg_utils.mtrf.matrices import banded_regularization

#from eeg_utils.mtrf.stats import *

from eeg_utils.utils.utils_statistics import bootstrap_permutation_test, jackknife_corr, jackknife_ttest

from eeg_utils.utils.fastcorr import *

import mne

mne.set_log_level('WARNING')

import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ttest_1samp, wilcoxon, shapiro, zscore
import matplotlib.pyplot as plt
from mtrf.model import TRF
from mne.preprocessing import ICA

import scipy.io as sio
import pandas as pd


def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """Applies Fisher z-transformation to Pearson's r values to normalize variance."""
    r_clipped = np.clip(r, -0.999, 0.999)
    return np.arctanh(r_clipped)

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
    z_base = fisher_z_transform(r_base)
    print(f'Best energy-only lambda: {opt_lambda_env}')

    # --- Full model (Energy + IC) ---
    if use_banded:
        print("Fitting Full Model with BANDED Regularization (Manual List Grid Search)...")
        best_r = -np.inf
        opt_lambda_full = [1.0, 1.0]
        n_epochs = len(stimuli_list)

        #reg_env = np.logspace(0, 4, 5)
        reg_env = np.logspace(-1, 3, 10)
        #reg_ic = np.logspace(-1, 3, 5)
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

        # Final LOO CV with the optimal lambdas
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
        # Standard Ridge Fallback
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
            
    z_full = fisher_z_transform(r_full)
    delta_z = z_full - z_base

    return z_full, z_base, delta_z, opt_lambda_full


def evaluate_with_circular_shift(stimuli_list, eeg_list, fs, lambda_val=1.0, n_perms=100, min_shift_sec=10.0, nested=True):#not currently used we preffered mismatched epochs, even if we wanted to use such shifting, linear shifting would be better for keeping auto-correlation intact (although it would cost some data)
    
    #Method 2: Full model vs. a Null distribution created via circular shifting.

    #Parameters:
    #- n_perms: number of random shifts to perform.
    #- min_shift_sec: minimum temporal shift in seconds to ensure alignment is broken.
    
    tmin, tmax = -0.1, 0.7
    k_folds = -1
    n_channels = eeg_list[0].shape[1]
    
    regularization = np.logspace(-1, 4, 6)
    
    trf = TRF(direction=1)

    # true full model
    print("Fitting True Full Model...")
    if nested :
        r_full, lambdas = nested_crossval(trf, stimuli_list, eeg_list, fs, tmin, tmax, regularization, k=k_folds, average=True, verbose=False)
        lambda_val = np.median(lambdas)
    r_full = crossval(trf, stimuli_list, eeg_list, fs, tmin, tmax, lambda_val, k=k_folds, average=False, verbose=False)
    if r_full.ndim==2:
        print("averaging r across trials")
        r_full=np.mean(r_full, axis=0)
    z_full = fisher_z_transform(r_full)

    # null distribution creation via circular shifting #TO DO : linear shifting could be used (for preventing autocorrelation breaking it is better than circular shifting) but we'll get very limited if we don't have many points #to counter this, phase randomization could be used to make a surrogate dataset : shuffling of the phases aloows us to get many samples for the null distribution.
    print(f"Generating Null Distribution with {n_perms} circular shifts...")
    z_null_permutations = np.zeros((n_perms, n_channels))
    min_shift_samples = int(min_shift_sec * fs)

    for p in range(n_perms):
        shuffled_stimuli = []

        for stim in stimuli_list:
            n_samples = stim.shape[0]
            stim_shuf = stim.copy()

            # random shift amount (at least min_shift, up to n_samples - min_shift)
            max_shift_samples = max(min_shift_samples + 1, n_samples - min_shift_samples)
            shift_amount = np.random.randint(min_shift_samples, max_shift_samples)

            # circularly shift (roll) only the IC column
            stim_shuf[:, 1] = np.roll(stim_shuf[:, 1], shift_amount)
            shuffled_stimuli.append(stim_shuf)

        if nested :
            r_null, lambdas = nested_crossval(trf, shuffled_stimuli, eeg_list, fs, tmin, tmax, regularization, k=k_folds, average=True, verbose=False)
            lambda_val = np.median(lambdas)
        # fit model on the circularly shifted data
        r_null = crossval(trf, shuffled_stimuli, eeg_list, fs, tmin, tmax, lambda_val, k=k_folds, average=False, verbose=False)
        if r_null.ndim==2:
            print("averaging r across trials")
            r_null=np.mean(r_null, axis=0)
        z_null_permutations[p, :] = fisher_z_transform(r_null)

    #null mean and enhancement
    z_null_mean = np.mean(z_null_permutations, axis=0)
    delta_z_null = z_full - z_null_mean

    #empirical p-value for each channel : how many times did a null model beat the true model?
    p_values = np.sum(z_null_permutations >= z_full, axis=0) / n_perms

    return z_full, z_null_mean, delta_z_null, p_values


def evaluate_with_mismatched_epochs(stimuli_list, target_list, fs, opt_lambda_full, n_perms=100, use_banded=False): 
    """
    Creates a null distribution by shuffling only the IC feature across epochs, 
    keeping the acoustic Envelope perfectly matched to the EEG to isolate IC impact.
    Handles length mismatches via circular wrapping (tiling) to avoid zero-padding artifacts.
    """
    tmin, tmax = -0.1, 0.7
    k_folds = -1
    n_channels = target_list[0].shape[1]
    n_epochs = len(stimuli_list)
    
    if n_epochs < 3:
        print("    [!] Not enough epochs to perform rigorous mismatching.")
        return np.zeros(n_channels), np.zeros((n_perms, n_channels))

    trf = TRF(direction=1)
    
    print(f"    Generating Null Distribution with {n_perms} Mismatched IC Permutations...")
    z_null_permutations = np.zeros((n_perms, n_channels))

    for p in range(n_perms):
        # generating a true derangement (no epoch matches itself)
        shuffled_indices = np.arange(n_epochs)
        while np.any(shuffled_indices == np.arange(n_epochs)):
            np.random.shuffle(shuffled_indices)
            
        # swap only the IC column
        mismatched_stimuli = []
        for i in range(n_epochs):
            stim_original = stimuli_list[i]
            stim_mismatch = stim_original.copy()
            
            # Grabbing IC from the mismatched epoch
            ic_wrong = stimuli_list[shuffled_indices[i]][:, 1]
            
            target_len = len(stim_original)
            source_len = len(ic_wrong)
            
            # Handling length differences via wrapping (tiling) instead of zero-padding
            if source_len >= target_len:
                # If the wrong IC is longer, we just truncate it
                new_ic = ic_wrong[:target_len]
            else:
                # If the wrong IC is shorter, we circularly repeat it until it covers the target length
                repeats_needed = int(np.ceil(target_len / source_len))
                new_ic = np.tile(ic_wrong, repeats_needed)[:target_len]
            
            # Replacing the original IC with the mismatched one
            stim_mismatch[:, 1] = new_ic
            mismatched_stimuli.append(stim_mismatch)

        # fitting the null model
        if isinstance(opt_lambda_full, list) or use_banded:
            r_null = crossval(
                trf, mismatched_stimuli, target_list, fs, tmin, tmax, 
                regularization=opt_lambda_full, bands=[1, 1], method='banded', 
                k=k_folds, average=False, verbose=False
            )
        else:
            r_null = crossval(
                trf, mismatched_stimuli, target_list, fs, tmin, tmax, 
                regularization=opt_lambda_full, k=k_folds, average=False, verbose=False
            )
            
        if r_null.ndim == 2:
            r_null = np.mean(r_null, axis=0)
            
        z_null_permutations[p, :] = fisher_z_transform(r_null)

    z_null_mean = np.mean(z_null_permutations, axis=0)
    
    return z_null_mean, z_null_permutations

##########################################################################################################################################
#Plotting topomap functions
##########################################################################################################################################



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

    # create MNE Info object
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


#######################################################################################################################
#Preprocessing and eeg preparation and alignment
#######################################################################################################################
#quality evaluation :

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


#general preprocessing :

def apply_bandpass_filter(raw: mne.io.BaseRaw, l_freq: float = 1.0, h_freq: float = 8.0) -> mne.io.BaseRaw:
    print(f"    -> Bandpass filtering ({l_freq} - {h_freq} Hz)...")
    # strictly picking EEG only so we don't accidentally filter and destroy the 'music' trigger channel
    picks = mne.pick_types(raw.info, eeg=True)
    raw.filter(l_freq, h_freq, picks=picks, fir_design='firwin', phase='zero', verbose=False)
    return raw
"""
def reject_bad_channels_by_correlation(raw: mne.io.BaseRaw, threshold_uv: float = 500.0) -> mne.io.BaseRaw: #actually not correlation based anymore
    print(f"    -> Checking for bad channels (amplitude > {threshold_uv} µV)...")
    threshold_v = threshold_uv * 1e-6 
    
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    
    #Ignore the first and last 2 seconds of the file where "filter ringing" causes fake spikes
    sfreq = raw.info['sfreq']
    trim_samples = int(2.0 * sfreq)
    
    #extracting the middle of the data to check amplitudes
    if raw.n_times > 2 * trim_samples:
        data = raw.get_data(picks=picks, start=trim_samples, stop=raw.n_times - trim_samples)
    else:
        data = raw.get_data(picks=picks)
        
    ptp_amps = np.ptp(data, axis=1)
    
    bad_idx = np.where(ptp_amps > threshold_v)[0]
    bad_ch_names = [raw.ch_names[picks[i]] for i in bad_idx]
    
    if bad_ch_names:
        if len(bad_ch_names) > len(picks) / 2:
            print(f"    -> WARNING: {len(bad_ch_names)} channels exceed threshold! Skipping interpolation.")
        else:
            print(f"    -> Interpolating {len(bad_ch_names)} noisy channels: {bad_ch_names}")
            raw.info['bads'].extend(bad_ch_names)
            raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        print("    -> All channels within acceptable amplitude limits.")
        
    return raw
"""
def find_bad_channels_by_standard_deviation(raw: mne.io.BaseRaw, std_multiplier: float = 3.0) -> mne.io.BaseRaw: 
    """
    Detects and interpolates bad EEG channels based on variance.
    Flags channels that are either completely flat (std ~ 0) or 
    excessively noisy (std > X times the mean std of all active channels).
    """
    print(f"    -> Checking for bad channels (flatlines or std > {std_multiplier}x mean)...")
    
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    
    # Ignore the first and last 2 seconds of the file where "filter ringing" causes fake spikes
    sfreq = raw.info['sfreq']
    trim_samples = int(2.0 * sfreq)
    
    # Extracting the middle of the data
    if raw.n_times > 2 * trim_samples:
        data = raw.get_data(picks=picks, start=trim_samples, stop=raw.n_times - trim_samples)
    else:
        data = raw.get_data(picks=picks)
        
    # Calculate the standard deviation for each channel
    stds = np.std(data, axis=1)
    
    # Flatlines
    # We use < 1e-10 instead of == 0 to account for microscopic floating-point noise
    flat_idx = np.where(stds < 1e-10)[0]
    flat_channels = [raw.ch_names[picks[i]] for i in flat_idx]
    
    if flat_channels:
        print(f"      - Found {len(flat_channels)} flatline channel(s): {flat_channels}")

    # Excessively noisy channels
    active_idx = np.where(stds >= 1e-10)[0]
    noisy_channels = []
    
    if len(active_idx) > 0:
        mean_std = np.mean(stds[active_idx])
        noise_threshold = std_multiplier * mean_std
        
        noisy_idx = np.where(stds > noise_threshold)[0]
        noisy_channels = [raw.ch_names[picks[i]] for i in noisy_idx]
        
        if noisy_channels:
            print(f"      - Found {len(noisy_channels)} noisy channel(s) exceeding threshold: {noisy_channels}")

    # combine and interpolate
    all_bad_channels = list(set(flat_channels + noisy_channels))
    
    if all_bad_channels:
        if len(all_bad_channels) > len(picks) / 2:
            print(f"    -> [!] WARNING: {len(all_bad_channels)} channels flagged! This implies global noise. Skipping interpolation.")
            #one could choose to strictly interpolate the flatlines here even if you skip the noisy ones
        else:
            print(f"    -> Interpolating {len(all_bad_channels)} bad channels: {all_bad_channels}")
            raw.info['bads'].extend(all_bad_channels)
            raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        print("    -> All channels have acceptable variance.")
        
    return raw

"""
def find_bad_channels_by_correlation(raw, corr_thresh=0.15): # Lowered to a realistic threshold
    print("    -> Detecting bad channels via neighbor correlation...")
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    
    #catching flatlines (variance = 0)
    stds = np.std(data, axis=1)
    flatline_idx = np.where(stds == 0)[0]
    
    bads_to_mark = [raw.ch_names[picks[i]] for i in flatline_idx]
    if bads_to_mark:
        print(f"      - Found flatline channels: {bads_to_mark}")

    #catching uncorrelated channels (drift / noise)
    valid_idx = np.where(stds > 0)[0]
    if len(valid_idx) > 1:
        valid_data = data[valid_idx, :]
        corr_matrix = np.corrcoef(valid_data)
        avg_corr = np.mean(corr_matrix, axis=1)
        
        poor_corr_idx = np.where(avg_corr < corr_thresh)[0]
        bads_to_mark.extend([raw.ch_names[picks[valid_idx[i]]] for i in poor_corr_idx])

    #safety net
    if bads_to_mark:
        bads_to_mark = list(set(bads_to_mark))
        # If it tries to reject more than 25% of the head, abort!
        if len(bads_to_mark) > len(picks) * 0.25:
            print(f"      - [!] WARNING: {len(bads_to_mark)} channels flagged! Aborting to prevent data loss.")
            # Only safely reject the true flatlines
            safe_bads = [raw.ch_names[picks[i]] for i in flatline_idx]
            raw.info['bads'].extend(safe_bads)
        else:
            print(f"      - Flagging channels for interpolation: {bads_to_mark}")
            raw.info['bads'].extend(bads_to_mark)
            
    return raw
"""
def remove_biological_artifacts_ica(raw: mne.io.BaseRaw, proxy_eog_ch: str = 'Fp1') -> mne.io.BaseRaw: #!this is currently done in the clean_eeg_mne_objects!
    """
    Uses ICA to automatically find and subtract both eye blinks and heartbeat artifacts.
    """
    print("    -> Running ICA for artifact removal (Ocular & Cardiac)...")
    
    #fitting ICA
    ica = ICA(n_components=15, random_state=42, max_iter='auto')
    ica.fit(raw, verbose=False)

    bads_to_exclude = []

    #removing ocular artifacts (blinks)
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=proxy_eog_ch, threshold=1.5, verbose=False)
        if eog_indices:
            bads_to_exclude.extend(eog_indices)
            print(f"      - Subtracted {len(eog_indices)} blink component(s).")
    except ValueError:
        print(f"      - Warning: Channel '{proxy_eog_ch}' not found for blink detection.")

    #removing cardiac artifacts (heartbeats) by automatically searching for 'ecg' channel types in the raw object
    if 'ecg' in raw.get_channel_types():
        try:
            #threshold='auto' relies on cross-trial phase tracking (CTPS), which is accurate for regular signals
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='ctps', verbose=False)
            if ecg_indices:
                bads_to_exclude.extend(ecg_indices)
                print(f"      - Subtracted {len(ecg_indices)} heartbeat component(s).")
        except Exception as e:
            print(f"      - Warning: Failed to detect cardiac components: {e}")
    else:
        print("      - No 'ecg' channel type found. Skipping heartbeat subtraction.")
        
    #removing duplicates just in case EOG and ECG flagged the same messy component
    ica.exclude = list(set(bads_to_exclude)) 
    ica.apply(raw, verbose=False)
    
    return raw

"""def remove_MRI_stationary_noise_with_SSP(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:

    #Computes Signal Space Projection (SSP) to remove the highest-variance 
    #stationary background noise (often continuous MRI machine hum).

    print("    -> Computing SSP to project out stationary MRI noise...")
    
    # Compute PCA-based projectors on continuous EEG data
    # We restrict it to 1 projection vector so we don't accidentally delete brain waves
    projs = mne.compute_proj_raw(raw, n_eeg=1, reject=None, verbose=False)
    
    if projs:
        # Add the projector to the raw object and apply it
        raw.add_proj(projs)
        raw.apply_proj(verbose=False)
        print("      - Projected out 1 spatial component.")
    else:
        print("      - No spatial projectors computed.")

    return raw"""



from meegkit.dss import dss_line

def extract_silent_eeg(raw, min_duration_sec=5.0):
    """
    Extracts purely silent epochs from the EEG recording to isolate background fMRI noise.
    """
    print("    -> Extracting silent blocks to isolate scanner noise...")
    fs = raw.info['sfreq']
    
    #finding where no music is playing
    music_data = raw.get_data(picks=['music'])[0]
    threshold = 1e-8 
    is_silent = music_data < threshold
    
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_data = raw.get_data(picks=eeg_picks)
    
    #look for where the silence turns on (1) and off (-1)
    changes = np.diff(is_silent.astype(int))
    silence_onsets = np.where(changes == 1)[0]
    silence_offsets = np.where(changes == -1)[0]
    
    #boundary cases (if it starts or ends in silence)
    if is_silent[0]:
        silence_onsets = np.insert(silence_onsets, 0, 0)
    if is_silent[-1]:
        silence_offsets = np.append(silence_offsets, len(is_silent) - 1)
        
    #collecting data from blocks longer than min_duration
    min_samples = int(min_duration_sec * fs)
    silent_chunks = []
    
    for start, end in zip(silence_onsets, silence_offsets):
        if (end - start) > min_samples:
            #shaving off the first and last 1 second to avoid the "click" of the audio turning on/off
            safe_start = start + int(1.0 * fs)
            safe_end = end - int(1.0 * fs)
            silent_chunks.append(eeg_data[:, safe_start:safe_end])
            
    if not silent_chunks:
        print("      [!] No long silent blocks found.")
        return None
        
    #stacking them all together into one matrix of pure silence
    pure_noise_matrix = np.hstack(silent_chunks)
    print(f"      - Extracted {pure_noise_matrix.shape[1] / fs:.2f} total seconds of pure silence.")
    
    return pure_noise_matrix

def remove_noise_subspace_via_pca(raw: mne.io.BaseRaw, silent_matrix: np.ndarray, subj_id: str, n_components: int = 1, outdir: str = "results") -> mne.io.BaseRaw:
    """
    Computes PCA on the pure silence matrix to isolate the spatial topography 
    of the fMRI scanner/background noise, then projects it out of the full recording.
    """
    print(f"    -> Computing PCA on {silent_matrix.shape[1]} samples of pure silence...")
    
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    
    silent_info = mne.pick_info(raw.info, sel=eeg_picks)
    silent_raw = mne.io.RawArray(silent_matrix, silent_info, verbose=False)
    
    #Computing the PCA spatial projectors (SSP)
    #n_eeg=1 removes the single largest component. You can increase to 2 if needed.
    projs = mne.compute_proj_raw(silent_raw, n_eeg=n_components, reject=None, verbose=False)
    
    #diagnostic plot: looking at what the noise looks like!
    if projs:
        os.makedirs(f"{outdir}/diagnostics/{subj_id}", exist_ok=True)
        fig = mne.viz.plot_projs_topomap(projs, info=silent_info, show=False)
        fig.suptitle(f"Removed fMRI Noise Topography (PCA)\nSubject {subj_id}")
        fig.savefig(f"{outdir}/diagnostics/{subj_id}/pca_noise_topography.png")
        plt.close(fig)
    
    print(f"    -> Projecting out the top {n_components} noise component(s) from the full recording...")
    raw_clean = raw.copy()
    raw_clean.add_proj(projs)
    raw_clean.apply_proj(verbose=False)
    
    return raw_clean

def remove_mri_harmonics_dss(raw: mne.io.BaseRaw, noise_freq: float) -> mne.io.BaseRaw:
    print(f"    -> Running DSS to remove periodic MRI noise at {noise_freq}Hz...")
    
    # extract data (shape must be: samples, channels)
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks).T 
    fs = raw.info['sfreq']
    
    # dss_line automatically finds the spatial components dominated by this frequency
    # and projects them out. nremove=1 or 2 is usually enough.
    clean_data, _ = dss_line(data, fline=noise_freq, sfreq=fs, nremove=2)
    
    # put clean data back into the raw object
    raw._data[picks] = clean_data.T
    return raw

def standardize_trial(eeg_matrix: np.ndarray) -> np.ndarray: #!this is currently done in the main analysis loop!
    """
    Step 4: Z-scores the EEG data across time to remove trial-to-trial DC offsets 
    and scale variance, preventing TRF bias.
    """
    print("    -> Z-scoring trial to remove baseline offsets...")
    # axis=0 standardizes across the time dimension for each channel independently
    return zscore(eeg_matrix, axis=0)

def clean_eeg_mne_object(raw, subj_id, outdir="results"):
    clean_raw = raw.copy()

    #labelling technical channels
    tech_ch_names = [
        'ft_valance', 'ft_arousal', 'ft_x', 'ft_y', 'ft_ghostvalence', 
        'ft_ghostarousal', 'trialtype', 'sams_valence', 
        'sams_arousal', 'sams_valencert', 'sams_arousalrt', 
        'nback_stimuli', 'nback_keypress'
    ]
    
    mapping = {ch: 'misc' for ch in tech_ch_names if ch in clean_raw.ch_names}
    if 'music' in clean_raw.ch_names:
        mapping['music'] = 'stim'
    if 'ECG' in clean_raw.ch_names:
        mapping['ECG'] = 'ecg'

    #mne.rename_channels(clean_raw.info, mapping={'FP1': 'Fp1', 'FP2': 'Fp2'})
    clean_raw.set_channel_types(mapping)

    #montage
    montage = mne.channels.make_standard_montage('standard_1005')
    clean_raw.set_montage(montage, match_case=False, on_missing='ignore')

    
    #downsampling to 64 Hz before filtering and ICA saves memory and time
    print("    -> Downsampling EEG to 64 Hz...")
    clean_raw.resample(64.0, npad='auto', verbose=False)
    
    #freq=?
    #clean_raw = remove_mri_harmonics_dss(clean_raw,freq)
    
    #silent_matrix = extract_silent_eeg(clean_raw, min_duration_sec=5.0)
    #projecting out the MRI noise
    #if silent_matrix is not None:
    #    clean_raw = remove_noise_subspace_via_pca(
    #        clean_raw, 
    #        silent_matrix, 
    #        subj_id, 
    #        n_components=1, # 2 can be chosen if the noise is highly complex
    #        outdir=outdir
    #    )

    #filtering
    clean_raw.filter(l_freq=1.0, h_freq=8.0, fir_design='firwin', verbose=False)

    #SSP for Stationary MRI Noise
    #clean_raw = remove_MRI_stationary_noise_with_SSP(clean_raw)

    #ICA for heartbeats and blinks
    picks_for_ica = mne.pick_types(clean_raw.info, eeg=True, ecg=True)
    ica_raw = clean_raw.copy().pick(picks=picks_for_ica)
    
    ica = ICA(n_components=15, random_state=42, method='fastica')
    ica.fit(ica_raw, verbose=False)

    eog_idx, _ = ica.find_bads_eog(clean_raw, ch_name='Fp1', threshold=2.0)
    ecg_idx, _ = ica.find_bads_ecg(clean_raw, method='correlation', threshold=1.5)
    
    ica.exclude = list(set(eog_idx + ecg_idx))
    print(f"    -> ICA: Ousting {len(ica.exclude)} components.")

    clean_raw = find_bad_channels_by_standard_deviation(clean_raw)
    if clean_raw.info['bads']:
        clean_raw.interpolate_bads(reset_bads=True, verbose=False)

    #diagnostic plots 
    plot_ica_diagnostics(clean_raw, ica, subj_id, outdir=outdir)

    ica.apply(clean_raw, verbose=False)
    
    return clean_raw
    


############################################################################
#specific to openneuro ds002725 :
"""
def decode_music_channel(voltage_value, feature_dir):
    
    #Decodes the voltage from the 'music' channel into the stimuli filename.
    
    number = int(round(voltage_value * 10**6 * 20))
    
    #strict matching by looking for "p3_" or "file_p3_" at the very start to AVOID GETTING CONFUSED BY THE op32 for instance !!!!!
    target_new = f"p{number}_"
    target_old = f"file_p{number}_"
    
    for f in os.listdir(feature_dir):
        if f.startswith(target_new) or f.startswith(target_old):
            return f
            
    print(f"Unknown track {number}")
    return None
"""
def decode_music_channel(voltage_value,feature_dir):
    number = int(round(voltage_value * 10**6 * 20))
    str_num = str(number)
    for f in os.listdir(feature_dir):
        if "p"+str_num in f :
            return f
    return None


def choose_stimulus_and_align_to_eeg(raw, env_dir, ic_dir, env_dt=0.004, ic_dt=0.7):
    """
    Finds all music onsets in the raw EEG, loads the corresponding envelope and IC features,
    interpolates them to the EEG sampling rate.
    
    Parameters:
    - raw: mne.io.Raw object of the EEG recording.
    - env_dir: Path to the directory where your envelope '.npy' files are saved.
    - ic_dir: Path to the directory where your IC '.npy' files are saved.
    - env_dt: The time step of your envelope features in seconds (e.g., 0.004).
    - ic_dt: The time step of your IC features in seconds (e.g., 0.7).
    
    Returns:
    - aligned_features: Continuous array of shape (eeg_length, 2) [Envelope, IC]
    """
    eeg_fs = raw.info['sfreq']
    print(eeg_fs)
    eeg_length = raw.n_times
    #TO DO : downsample eeg to 100Hz, 64 Hz or even 32Hz
    #band-pass filter, for instance between 1 and 30 Hz or even better between 1 and 8 Hz DONE in clean_mne_eeg_object
    
    # Extract only the 'music' channel
    music_data, times = raw.get_data(picks=['music'], return_times=True)
    music_data = music_data[0] 

    # finding where the music actually starts (voltage goes from 0 to > 0.5V)
    threshold = 1e-8 
    is_playing = music_data > threshold
    onsets = np.where(np.diff(is_playing.astype(int)) == 1)[0] + 1

    # 2 columns: 0 for Envelope, 1 for IC
    aligned_features = np.zeros((eeg_length, 2))
    valid_stimuli_found = False

    print(" --- Trial Alignments --- ")
    for onset_idx in onsets:
        start_time_sec = times[onset_idx]
        stable_voltage = music_data[onset_idx + 10] 
        
        env_filename = decode_music_channel(stable_voltage, env_dir)
        
        if env_filename is None:
            print(f"stimulus not found in {raw} at {start_time_sec}")
            continue
            
        # Determine paths (replacing 'envelope' with 'nll' for the IC file)
        ic_filename = env_filename.replace("envelope", "nll")
        
        env_path = os.path.join(env_dir, env_filename)
        ic_path = os.path.join(ic_dir, ic_filename)
        
        if not os.path.exists(env_path):
            print(f"  [!] Warning: Envelope file {env_path} not found. Skipping.")
            continue
        if not os.path.exists(ic_path):
            print(f"  [!] Warning: IC file {ic_path} not found. Skipping.")
            continue
            
        # Load the features and ensure they are flat 1D arrays
        env_data = np.squeeze(np.load(env_path))
        ic_data = np.squeeze(np.load(ic_path))
        
        # independent time vectors based on their specific sampling rates
        env_time = np.arange(len(env_data)) * env_dt
        ic_time = np.arange(len(ic_data)) * ic_dt

        #print(f"  [DEBUG] Calculated Env length: {env_time[-1]:.2f} seconds")
        #print(f"  [DEBUG] Calculated IC length: {ic_time[-1]:.2f} seconds")
        
        # maximum duration in seconds to know how many EEG samples to fill
        max_duration_sec = max(env_time[-1], ic_time[-1]) if len(env_time)>0 and len(ic_time)>0 else 0
        #max_duration_sec = env_time[-1] if len(env_time) > 0 else 0
        n_eeg_samples = int(np.ceil(max_duration_sec * eeg_fs))
        eeg_time = np.arange(n_eeg_samples) / eeg_fs
        
        #interpolating envelope (linear is usually best for continuous audio energy)
        env_interpolator = interp1d(
            env_time, env_data, kind='linear', bounds_error=False, fill_value=0.0
        )
        env_aligned = env_interpolator(eeg_time)
        
        #interpolating IC 
        ic_interpolator = interp1d(
            ic_time, ic_data, kind='previous', bounds_error=False, fill_value=0.0
        ) #TO DO : decide what is best here between linear and previous
        ic_aligned = ic_interpolator(eeg_time)
        
        # Determine insertion bounds to prevent array overflow at the end of the recording
        end_idx = min(onset_idx + n_eeg_samples, eeg_length)
        length_to_insert = end_idx - onset_idx
        
        # Place the interpolated stimuli exactly at the trigger onset
        aligned_features[onset_idx:end_idx, 0] = env_aligned[:length_to_insert]
        aligned_features[onset_idx:end_idx, 1] = ic_aligned[:length_to_insert]
        
        valid_stimuli_found = True
        print(f"  Mapped: {start_time_sec:.2f}s -> {env_filename},{ic_filename}")
        
    if not valid_stimuli_found:
        raise ValueError("No valid stimuli were found or loaded for this EEG recording.")
        
    return aligned_features



def run_openneuro_dataset_analysis(bids_root_dir, env_dir, ic_dir, env_dt=0.004, ic_dt=0.7, use_banded=False, outdir="results"):
    """
    Loops through an OpenNeuro BIDS dataset, aligns data, runs Banded TRF, 
    and computes robust group-level non-parametric summary statistics.
    """
    subject_folders = sorted(glob.glob(os.path.join(bids_root_dir, "sub-*")))
    if not subject_folders:
        raise ValueError(f"No BIDS sub-* folders found in {bids_root_dir}")
        
    all_delta_z = []
    all_z_base = []
    
    channel_names = None
    eeg_fs = None 
    
    # 10-05 channels
    eeg_only_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'
    ]
    

    all_behav_delta_z = []
    all_arousal_weights = []
    
    behav_channel_names = ['ft_valance', 'ft_arousal']

    cpt_flat_chunks=0

    for subj_path in subject_folders:
        subj_id = os.path.basename(subj_path)
        print(f"\n=============================")
        print(f" Processing {subj_id}...")
        print(f"=============================")
        
        eeg_list = []
        behav_list = []
        stimuli_list = []

        
        edf_files = sorted(glob.glob(os.path.join(subj_path, 'eeg', '*.edf')))
        if not edf_files:
            print(f"No EEG data found for {subj_id}, skipping...")
            continue
            
        for edf_file in edf_files:
            if "genMusic" in edf_file:
                continue

            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            if eeg_fs is None:
                eeg_fs = raw.info['sfreq']
            
            try:
                clean_raw = clean_eeg_mne_object(raw, subj_id, outdir)
                    
                #updating eeg_fs to the new resampled rate
                eeg_fs = clean_raw.info['sfreq']

                #aligning stimuli using the clean MNE object
                aligned_stim = choose_stimulus_and_align_to_eeg(
                    raw=clean_raw, 
                    env_dir=env_dir, 
                    ic_dir=ic_dir, 
                    env_dt=env_dt, 
                    ic_dt=ic_dt
                )

                #picking only EEG channels for the TRF target
                eeg_picks = mne.pick_types(clean_raw.info, eeg=True)
                
                if channel_names is None:
                    channel_names = [clean_raw.ch_names[i] for i in eeg_picks]

                #eliminating NaNs
                #shape: (n_samples, n_eeg_channels)
                raw_eeg_trial = clean_raw.get_data(picks=eeg_picks).T
                raw_eeg_trial = np.nan_to_num(raw_eeg_trial, nan=0.0)

                #extracting behavioral data (valence and arousal)
                behav_picks = mne.pick_channels(clean_raw.ch_names, behav_channel_names)
                if len(behav_picks) > 0:
                    raw_behav_trial = np.nan_to_num(clean_raw.get_data(picks=behav_picks).T, nan=0.0)
                else:
                    raw_behav_trial = None

                #chunking for "data augmentation" 
                valid_indices = np.where(aligned_stim[:, 0] > 0)[0]
                #valid_indices = np.where(aligned_stim[:, 0] > -1)[0]
                if len(valid_indices) == 0:
                    continue

                #gap_tolerance_samples = int(5.0 * eeg_fs) 
                #split_points = np.where(np.diff(valid_indices) > gap_tolerance_samples)[0] + 1
                split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
                valid_blocks = np.split(valid_indices, split_points)
                
                for block in valid_blocks:
                    if len(block) > (10 * eeg_fs):
                        chunk_size = int(30 * eeg_fs) 
                        for i in range(0, len(block), chunk_size):
                            chunk_idx = block[i : i + chunk_size]

                            if len(chunk_idx) > (15 * eeg_fs):
                                #chunks
                                raw_chunk_stim = aligned_stim[chunk_idx, :]
                                raw_chunk_eeg = raw_eeg_trial[chunk_idx, :]
                                
                                #checking if EEG has flatlined
                                if np.any(np.std(raw_chunk_eeg, axis=0) == 0):
                                    continue
                                    
                                #checking is the acoustic envelope has flatlined
                                #if np.std(raw_chunk_stim[:, 0]) == 0:
                                #    continue
                                
                                #scipy's zscore turns flat columns into NaNs. np.nan_to_num catches those NaNs and converts them to 0.0
                                clean_stim = np.nan_to_num(zscore(raw_chunk_stim, axis=0), nan=0.0)
                                ic_variance = np.var(clean_stim[:, 1])
                                if ic_variance == 0.0:
                                    cpt_flat_chunks+=1
                                    print(f"  [DEBUG] IC column flatlined! This is the {cpt_flat_chunks}th we lose this way.")
                                clean_eeg = np.nan_to_num(zscore(raw_chunk_eeg, axis=0), nan=0.0)

                                stimuli_list.append(clean_stim)
                                eeg_list.append(clean_eeg)

                                if raw_behav_trial is not None:
                                    raw_chunk_behav = raw_behav_trial[chunk_idx, :]
                                    clean_behav = np.nan_to_num(zscore(raw_chunk_behav, axis=0), nan=0.0)
                                    behav_list.append(clean_behav)
            except ValueError as e:
                print(f"  [!] {e} Skipping EDF.")
                continue
            
        # ====================================================================
        # Subject-Level TRF Pipeline
        if not stimuli_list:
            continue
            
        try:
            print(f"  -> Fitting direct models (Cross-Validation on {len(stimuli_list)} epochs)...")

            z_full, z_base, delta_z, opt_lambda_full = compare_direct_models(stimuli_list, eeg_list, eeg_fs, use_banded=use_banded)
            
            delta_z = np.nan_to_num(delta_z, nan=0.0)
            z_base = np.nan_to_num(z_base, nan=0.0)
            all_delta_z.append(delta_z)
            all_z_base.append(z_base)
            print(f"[{subj_id}] Mean EEG channel IC predictive enhancement: {np.mean(delta_z):.4f}")

            #plotting topomap according to the value of predictive enhancement obtained from IC in each channel
            eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
            valid_eeg_names = [channel_names[i] for i in eeg_indices]
            scalp_scores = delta_z[eeg_indices]

            os.makedirs(outdir, exist_ok=True)

            fig = plot_static_score_topomap(
                scores=scalp_scores, 
                ch_names=valid_eeg_names, 
                subj_id=subj_id,
                score_type="Δz (Full - Baseline)",
                cmap="Reds",
                vmin=0.0  
            )
            fig.savefig(f"{outdir}/topomap_{subj_id}_delta_z.png", dpi=300)
            plt.close(fig)

            if behav_list:
                print(f"  -> Fitting models for Continuous Emotion (Valence/Arousal)...")
                
                # Capture the optimal lambda (opt_lambda_b) instead of using '_'
                z_full_b, z_base_b, delta_z_b, opt_lambda_b = compare_direct_models(stimuli_list, behav_list, eeg_fs, use_banded=use_banded)
                
                delta_z_b = np.nan_to_num(delta_z_b, nan=0.0)
                all_behav_delta_z.append(delta_z_b)
                print(f"[{subj_id}] Emotion predictive enhancement: {np.mean(delta_z_b):.4f}")
                
                # -- Add this to plot the weights! --
                print(f"  -> Extracting temporal weights for Arousal...")
                
                behav_trf = TRF(direction=1)
                
                # Check if opt_lambda_b is banded (a list/array) or standard (a scalar)
                if isinstance(opt_lambda_b, (list, np.ndarray)):
                    behav_trf.train(
                        stimuli_list,  
                        behav_list,    
                        eeg_fs, 
                        tmin=-0.1, 
                        tmax=0.7,      
                        regularization=opt_lambda_b,
                        bands=[1, 1]
                    )
                else:
                    behav_trf.train(
                        stimuli_list, 
                        behav_list, 
                        eeg_fs, 
                        tmin=-0.1, 
                        tmax=0.7,      
                        regularization=opt_lambda_b
                    )

                fig_behav = plot_behavioral_trf_weights(
                    trf_model=behav_trf, 
                    fs=eeg_fs, 
                    tmin=-0.1, 
                    tmax=0.7,         
                    subj_id=subj_id
                )
                fig_behav.savefig(f"{outdir}/arousal_weights_{subj_id}.png", dpi=300)
                plt.close(fig_behav)

                subj_arousal_weights = behav_trf.weights[1, :, 1]
                all_arousal_weights.append(subj_arousal_weights)
            
            """# Fit final full model with banded constraints
            print(f"  -> Fitting final Full Model to extract weights...")
            all_stimuli_concat = np.vstack(stimuli_list)
            all_eeg_concat = np.vstack(eeg_list)

            final_trf = TRF(direction=1)

            #training specifically with the banded framework
            final_trf.train(
                all_stimuli_concat, 
                all_eeg_concat, 
                eeg_fs, 
                tmin=-0.1, 
                tmax=0.7, 
                regularization=opt_lambda_full, # Passed as a list [l_env, l_ic]
                bands=[1, 1],                   # Tell mTRFpy the dimensions of the bands
                method='banded'                 
            )"""
            
            #fitting one final full model for each subject on all trials for each channel to get the weights for each lag 
            print(f"  -> Fitting final full model to extract weights")
            
            final_trf = TRF(direction=1)
            
            # If we used banded, pass the bands argument. Otherwise, standard.
            if isinstance(opt_lambda_full, list):
                final_trf.train(
                    stimuli_list, 
                    eeg_list, 
                    eeg_fs, 
                    tmin=-0.1, 
                    tmax=0.7, 
                    regularization=opt_lambda_full,
                    bands=[1, 1]
                )
            else:
                final_trf.train(
                    stimuli_list, 
                    eeg_list, 
                    eeg_fs, 
                    tmin=-0.1, 
                    tmax=0.7, 
                    regularization=opt_lambda_full
                )

            sliced_weights = final_trf.weights[:, :, eeg_indices]
            final_trf.weights = sliced_weights

            fig = plot_participant_trf_topomaps(
                trf_model=final_trf,
                ch_names=valid_eeg_names, 
                fs=eeg_fs,
                tmin=-0.1,
                subj_id=subj_id,
                feature_idx=1,
                feature_name="Information Content (IC)",
                times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 
            )
            fig.savefig(f"{outdir}/topomap_{subj_id}_IC_weights.png", dpi=300)
            plt.close(fig)

        except Exception as e:
            print(f"Failed to process {subj_id} TRF: {e}")
            continue

    if not all_delta_z:
        raise RuntimeError("No subjects were successfully processed.")

    # ====================================================================
    # Group-Level Statistics 

    print("\n==========================================")
    print(" Computing Group-Level Statistics...")
    print("==========================================")

    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]
    
    #baseline (envelope) predictions z-transformed pearson correlation coefficients 
    group_z_base = np.stack(all_z_base, axis=0) 
    
    #subset to the 30 scalp electrodes
    eeg_group_z_base = group_z_base[:, eeg_indices]
    
    mean_z_base = np.mean(eeg_group_z_base, axis=0)
    std_z_base = np.std(eeg_group_z_base, axis=0)
    
    #non-parametric Jackknife testing on the 30 scalp electrodes
    p_values_base = []
    for ch in range(eeg_group_z_base.shape[1]):
        try:
            _, p = jackknife_ttest(eeg_group_z_base[:, ch], alternative='greater')
        except Exception:
            #fallback to standard Wilcoxon if jackknife fails
            res = wilcoxon(eeg_group_z_base[:, ch], alternative='greater') 
            p = res.pvalue
        p_values_base.append(p)
        
    p_values_base = np.array(p_values_base)
    
    #FDR correction to the 30 valid p-values
    reject_null_base, p_values_fdr_base = mne.stats.fdr_correction(p_values_base, alpha=0.05, method='indep')
    significant_channels_base = np.where(reject_null_base)[0]
    
    print(f"Overall Mean Scalp Envelope Prediction (z_base): {np.mean(mean_z_base):.4f}")
    print(f"Number of channels with significant envelope prediction: {len(significant_channels_base)} / {len(valid_eeg_names)}")
    
    if len(significant_channels_base) > 0:
        print("\nSignificant Envelope Channels (FDR Corrected):")
        for ch_idx in significant_channels_base:
            print(f" - {valid_eeg_names[ch_idx]}: z_base = {mean_z_base[ch_idx]:.4f} (p = {p_values_fdr_base[ch_idx]:.4f})")

            
    #predicitve enhancement obtained from IC on each channel for all pariticpants
    group_delta_z = np.stack(all_delta_z, axis=0) 
    
    #shape becomes (18 subjects, 30 channels)
    eeg_group_delta_z = group_delta_z[:, eeg_indices]
    
    mean_delta_z = np.mean(eeg_group_delta_z, axis=0)
    std_delta_z = np.std(eeg_group_delta_z, axis=0)
    
    #non-parametric Jackknife testing on the 30 scalp electrodes
    p_values = []
    for ch in range(eeg_group_delta_z.shape[1]):
        try:
            _, p = jackknife_ttest(eeg_group_delta_z[:, ch], alternative='greater')
        except Exception:
            #Fallback to standard Wilcoxon if jackknife fails
            res = wilcoxon(eeg_group_delta_z[:, ch], alternative='greater') #NB: a simple t test could also work but only if the normality condition is verified
            p = res.pvalue
        p_values.append(p)
        
    p_values = np.array(p_values)
    
    #FDR correction to the 30 valid p-values
    reject_null, p_values_fdr = mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')
    significant_channels = np.where(reject_null)[0]
    
    print(f"Total Subjects Processed: {eeg_group_delta_z.shape[0]}")
    print(f"Overall Mean Scalp Enhancement: {np.mean(mean_delta_z):.4f}")
    print(f"Number of EEG channels with significant IC enhancement: {len(significant_channels)} / {len(valid_eeg_names)}")
    
    if len(significant_channels) > 0:
        print("\nSignificant Channels (FDR Corrected):")
        for ch_idx in significant_channels:
            print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr[ch_idx]:.4f})")

    # ====================================================================
    # ROI-averaged statistical testing

    print("\n==========================================")
    print(" Computing ROI-Averaged Statistics...")
    print("==========================================")

    
    # Region of Interest (ROI) - bilateral fronto-central-temporal cluster for broad auditory processing
    roi_names = ['Fz', 'Cz', 'FC1', 'FC2', 'C3', 'C4', 'FC5', 'FC6', 'T7', 'T8', 'CP5', 'CP6'] 
    
    #indices for these specific channels
    roi_indices = [valid_eeg_names.index(ch) for ch in roi_names if ch in valid_eeg_names]

    if roi_indices:
        # eeg_group_delta_z is shape (n_subjects, n_channels)
        roi_delta_z = eeg_group_delta_z[:, roi_indices]

        #averaging spatially across the ROI for each subject -> shape (n_subjects,)
        subj_roi_averages = np.mean(roi_delta_z, axis=1)
        mean_roi_enhancement = np.mean(subj_roi_averages)

        #checking if the distribution of ROI averages is normal
        stat_shap, p_shapiro = shapiro(subj_roi_averages)
        
        print(f"ROI Channels Used: {[valid_eeg_names[i] for i in roi_indices]}")
        print(f"Mean ROI Enhancement: {mean_roi_enhancement:.4f}")
        
        #chooosing the appropriate test based on normality
        if p_shapiro < 0.05:
            print("  -> Normality violated (Shapiro-Wilk p < 0.05). Using Non-Parametric Wilcoxon Test.")
            res = wilcoxon(subj_roi_averages, alternative='greater')
            p_val_roi = res.pvalue
            test_name = "Wilcoxon Signed-Rank"
        else:
            print("  -> Data is normally distributed. Using 1-Sample T-Test.")
            res = ttest_1samp(subj_roi_averages, popmean=0.0, alternative='greater')# 1-sample t-test on the difference scores (delta_z), which is mathematically identical to a paired t-test of Full vs Baseline.
            p_val_roi = res.pvalue
            test_name = "1-Sample T-Test"

        if p_val_roi < 0.05:
            print(f"ROI {test_name}: p = {p_val_roi:.4f} *** SIGNIFICANT ***")
        else:
            print(f"ROI {test_name}: p = {p_val_roi:.4f} (Not Significant)")

    if all_behav_delta_z:
        print("\n==========================================")
        print(" Computing Group-Level Behavioural Statistics...")
        print("==========================================")
        
        # Shape becomes (n_subjects, 2) [Valence, Arousal]
        group_behav_delta_z = np.stack(all_behav_delta_z, axis=0) 
        
        mean_behav_delta_z = np.mean(group_behav_delta_z, axis=0)
        std_behav_delta_z = np.std(group_behav_delta_z, axis=0)
        
        # Non-parametric testing for the 2 behavioral channels
        p_values_behav = []
        for ch in range(group_behav_delta_z.shape[1]):
            try:
                _, p = jackknife_ttest(group_behav_delta_z[:, ch], alternative='greater')
            except Exception:
                # Fallback to standard Wilcoxon if jackknife fails
                res = wilcoxon(group_behav_delta_z[:, ch], alternative='greater') 
                p = res.pvalue
            p_values_behav.append(p)
            
        p_values_behav = np.array(p_values_behav)
        
        # FDR correction for the 2 p-values
        reject_null_behav, p_values_fdr_behav = mne.stats.fdr_correction(p_values_behav, alpha=0.05, method='indep')
        
        print(f"Total subjects processed for behavioural testing: {group_behav_delta_z.shape[0]}")
        print("Continuous Behavioural data predictive enhancement (IC over envelope baseline):")
        
        for ch_idx, behav_name in enumerate(behav_channel_names):
            sig_flag = "*** SIGNIFICANT ***" if reject_null_behav[ch_idx] else "(Not Significant)"
            print(f" - {behav_name.upper()}: Δz = {mean_behav_delta_z[ch_idx]:.4f} (p_fdr = {p_values_fdr_behav[ch_idx]:.4f}) {sig_flag}")

        if all_arousal_weights:
            fig_group_behav = plot_group_behavioral_weights(
                all_weights=all_arousal_weights, 
                fs=eeg_fs, 
                tmin=-0.1, 
                tmax=0.7
            )
            fig_group_behav.savefig(f"{outdir}/group_level_arousal_weights.png", dpi=300)
            plt.close(fig_group_behav)
            print(f"Saved group-level behavioral plot to {outdir}/group_level_arousal_weights.png")
            
    #adding behavioral results to the final return dictionary
    return {
        "mean_delta_z": mean_delta_z,
        "std_delta_z": std_delta_z,
        "p_values_raw": p_values,
        "p_values_fdr": p_values_fdr,
        "significant_mask": reject_null,
        "channel_names": valid_eeg_names,
        "behav_mean_delta_z": mean_behav_delta_z if all_behav_delta_z else None,
        "behav_p_values_fdr": p_values_fdr_behav if all_behav_delta_z else None,
        "behav_names": behav_channel_names if all_behav_delta_z else None
    }




#################################################################################################################################################################
#SONG dataset treatment

def prepare_song_trial(eeg_matrix, env_path, ic_path, eeg_fs, env_dt=0.015625, ic_dt=0.7):
    """
    Takes the pre-cut EEG matrix, loads the standalone Envelope (.npy) and IC (.csv OR .npy), 
    and perfectly aligns everything to the EEG sampling rate.
    """
    # 1. Load Acoustic Envelope (always .npy)
    env_data = np.squeeze(np.load(env_path))
    env_time = np.arange(len(env_data)) * env_dt
    
    eeg_length = eeg_matrix.shape[0]
    eeg_time = np.arange(eeg_length) / eeg_fs
    
    # 2. Auto-Detect and Load Information Content (IC)
    if ic_path.endswith('.csv'):
        # CSV FORMAT 
        ic_df = pd.read_csv(ic_path)
        ic_time = ic_df['time'].values
        ic_data = ic_df['nll_0.000'].fillna(0.0).values
        #linear interpolation 
        ic_interpolator = interp1d(ic_time, ic_data, kind='linear', bounds_error=False, fill_value=0.0)
        
    elif ic_path.endswith('.npy'):
        #NPY FORMAT
        ic_data = np.squeeze(np.load(ic_path))
        ic_time = np.arange(len(ic_data)) * ic_dt
        ic_interpolator = interp1d(ic_time, ic_data, kind='linear', bounds_error=False, fill_value=0.0)
        
    else:
        raise ValueError(f"Unsupported IC file format: {ic_path}. Must be .csv or .npy")

    #aligning the time series
    ic_aligned = ic_interpolator(eeg_time)
    
    env_interpolator = interp1d(env_time, env_data, kind='linear', bounds_error=False, fill_value=0.0)
    env_aligned = env_interpolator(eeg_time)
    
    #combining and Z-scoring
    aligned_stim = np.column_stack((env_aligned, ic_aligned))
    clean_stim = np.nan_to_num(zscore(aligned_stim, axis=0))
    clean_eeg = np.nan_to_num(zscore(eeg_matrix, axis=0))
    
    return clean_stim, clean_eeg


 

def run_song_dataset_analysis(mat_dir, env_dir, csv_dir, eeg_fs=None, env_dt=0.015625, ic_dt=0.7, use_banded=False, n_perms=None, outdir="results"):
    
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "pre_dataSub*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No matching .mat files found in '{mat_dir}'.")
    
    csv_files = sorted(
        glob.glob(os.path.join(csv_dir, "ic_*_song.csv")) + 
        glob.glob(os.path.join(csv_dir, "*_song.npy"))
    )

    if not csv_files:
        raise FileNotFoundError(f"No matching IC files (.csv or .npy) found in '{csv_dir}'.")
    
    all_delta_z = []
    all_z_base = []
    all_eeg_null_perms = []
    
    channel_names = None

    eeg_only_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
    ]
    
    for mat_file in mat_files:
        subj_id = os.path.basename(mat_file).split('.')[0]
        print(f"\n=============================")
        print(f" Processing {subj_id}...")
        
        #loading preprocessed subject data
        mat_data = sio.loadmat(mat_file)
        eeg_struct = mat_data['eeg'][0, 0]
        
        trial_fs = int(eeg_struct['fs'].item())
        if eeg_fs is None:
            eeg_fs = trial_fs
            
        eeg_trials = eeg_struct['data'][0]
        
        # extracting channel names from the MATLAB struct on the first pass
        if channel_names is None:
            chanlocs = eeg_struct['chanlocs'][0]
            channel_names = [ch['labels'][0] for ch in chanlocs]
            print(f"    -> Extracted {len(channel_names)} channels from MAT file.")

            eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
            valid_eeg_names = [channel_names[i] for i in eeg_indices]
        
        eeg_list = []
        stimuli_list = []
        
        for trial_idx in range(len(csv_files)):
            trial_eeg = eeg_trials[trial_idx] 
            
            if trial_eeg.size == 0 or max(trial_eeg.shape) < (eeg_fs * 2):
                print(f"  [!] Warning: Trial {trial_idx} is empty or too short (< 2s). Skipping.")
                continue
                
            trial_eeg = np.squeeze(trial_eeg)
            if len(trial_eeg.shape) < 2:
                print(f"  [!] Warning: Trial {trial_idx} has invalid dimensions. Skipping.")
                continue
                
            if trial_eeg.shape[0] < trial_eeg.shape[1]:
                trial_eeg = trial_eeg.T
                
            current_ic_path = csv_files[trial_idx]
            ic_filename = os.path.basename(current_ic_path)
            
            base_name = ic_filename.replace('ic_', '').rsplit('.', 1)[0]
            env_filename = base_name + '.npy' 
            env_path = os.path.join(env_dir, env_filename)
            
            if not os.path.exists(env_path):
                print(f"  [!] Warning: Envelope {env_filename} not found. Skipping trial.")
                continue
                
            clean_stim, clean_eeg = prepare_song_trial(
                eeg_matrix=trial_eeg, 
                env_path=env_path, 
                ic_path=current_ic_path, 
                eeg_fs=eeg_fs, 
                env_dt=env_dt,
                ic_dt=ic_dt 
            )
            
            stimuli_list.append(clean_stim)
            eeg_list.append(clean_eeg)
            
        if not stimuli_list:
            continue
            
        # TRF Cross-Validation
        print(f"  -> Fitting direct models for {subj_id} on {len(stimuli_list)} epochs...")
        z_full, z_base, delta_z, opt_lambda_full = compare_direct_models(stimuli_list, eeg_list, eeg_fs, use_banded=use_banded)
        
        if n_perms:
            _, z_null_perms = evaluate_with_mismatched_epochs(
                stimuli_list, eeg_list, eeg_fs, opt_lambda_full, n_perms=n_perms, use_banded=use_banded
            )
            all_eeg_null_perms.append(z_null_perms)
        
        delta_z = np.nan_to_num(delta_z, nan=0.0)
        z_base = np.nan_to_num(z_base, nan=0.0)
        
        all_delta_z.append(delta_z)
        all_z_base.append(z_base)
        print(f"[{subj_id}] Mean channel enhancement: {np.mean(delta_z):.4f}")
        
        # --- participant level topomaps ---
        scalp_scores = delta_z[eeg_indices]
        os.makedirs(outdir, exist_ok=True)

        fig_topo = plot_static_score_topomap(
            scores=scalp_scores, 
            ch_names=valid_eeg_names, 
            subj_id=subj_id,
            score_type="Δz (Full - Baseline)",
            cmap="Reds",
            vmin=0.0  
        )
        fig_topo.savefig(f"{outdir}/topomap_{subj_id}_delta_z.png", dpi=300)
        plt.close(fig_topo)
        
        #fitting one final full model on all trials
        print(f"  -> Fitting final full model to extract weights")
        final_trf = TRF(direction=1)
            
        if isinstance(opt_lambda_full, list):
            final_trf.train(
                stimuli_list, eeg_list, eeg_fs, tmin=-0.1, tmax=0.7, 
                regularization=opt_lambda_full, bands=[1, 1]
            )
        else:
            final_trf.train(
                stimuli_list, eeg_list, eeg_fs, tmin=-0.1, tmax=0.7, 
                regularization=opt_lambda_full
            )

        sliced_weights = final_trf.weights[:, :, eeg_indices]
        final_trf.weights = sliced_weights

        fig = plot_participant_trf_topomaps(
            trf_model=final_trf,
            ch_names=valid_eeg_names, 
            fs=eeg_fs,
            tmin=-0.1,
            subj_id=subj_id,
            feature_idx=1,
            feature_name="Information Content (IC)",
            times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 
        )
        fig.savefig(f"{outdir}/topomap_song_{subj_id}_IC_weights.png", dpi=300)
        plt.close(fig)
        
    if not all_delta_z:
        raise RuntimeError("No subjects were successfully processed.")
        
    # ====================================================================
    # Group-Level Statistics 

    print("\n==========================================")
    print(" Computing Group-Level Statistics...")
    print("==========================================")

    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]
    
    group_z_base = np.stack(all_z_base, axis=0) 
    eeg_group_z_base = group_z_base[:, eeg_indices]
    mean_z_base = np.mean(eeg_group_z_base, axis=0)
    std_z_base = np.std(eeg_group_z_base, axis=0)
    
    p_values_base = []
    for ch in range(eeg_group_z_base.shape[1]):
        try:
            _, p = jackknife_ttest(eeg_group_z_base[:, ch], alternative='greater')
        except Exception:
            res = wilcoxon(eeg_group_z_base[:, ch], alternative='greater') 
            p = res.pvalue
        p_values_base.append(p)
        
    p_values_base = np.array(p_values_base)
    reject_null_base, p_values_fdr_base = mne.stats.fdr_correction(p_values_base, alpha=0.05, method='indep')
    significant_channels_base = np.where(reject_null_base)[0]
    
    print(f"Overall Mean Scalp Envelope Prediction (z_base): {np.mean(mean_z_base):.4f}")
    if len(significant_channels_base) > 0:
        print("\nSignificant Envelope Channels (FDR Corrected):")
        for ch_idx in significant_channels_base:
            print(f" - {valid_eeg_names[ch_idx]}: z_base = {mean_z_base[ch_idx]:.4f} (p = {p_values_fdr_base[ch_idx]:.4f})")
            
    group_delta_z = np.stack(all_delta_z, axis=0) 
    eeg_group_delta_z = group_delta_z[:, eeg_indices]
    mean_delta_z = np.mean(eeg_group_delta_z, axis=0) #mean on all subjects for each channel
    std_delta_z = np.std(eeg_group_delta_z, axis=0)

    # --- permutation testing logic if enabled ---
    if n_perms:
        print("  -> Using Permutation Testing for Group IC stats...")
        group_eeg_null_perms = np.stack(all_eeg_null_perms, axis=0)[:, :, eeg_indices]
        base_expanded = eeg_group_z_base[:, np.newaxis, :]
        null_delta_z_matrix = group_eeg_null_perms - base_expanded
        group_null_delta_z_dist = np.mean(null_delta_z_matrix, axis=0) 
        
        p_values = np.sum(group_null_delta_z_dist >= mean_delta_z, axis=0) / n_perms


    #old statistical way : without permutations
    print("  -> Using Jackknife/Wilcoxon Testing for Group IC stats...")
    p_values_no_perm = []
    for ch in range(eeg_group_delta_z.shape[1]):
        try:
            _, p = jackknife_ttest(eeg_group_delta_z[:, ch], alternative='greater')
        except Exception:
            res = wilcoxon(eeg_group_delta_z[:, ch], alternative='greater') 
            p = res.pvalue
        p_values_no_perm.append(p)
    ####

    if n_perms:
        p_values = np.array(p_values)
        reject_null, p_values_fdr = mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')
        significant_channels = np.where(reject_null)[0]

        #subtracting the mean null distribution enhancement from the true enhancement
        mean_null_floor = np.mean(group_null_delta_z_dist, axis=0)
        corrected_delta_z = mean_delta_z - mean_null_floor
        
        print(f"\nTotal Subjects Processed: {eeg_group_delta_z.shape[0]}")
        print(f"Overall Mean Scalp Enhancement: {np.mean(mean_delta_z):.4f}")
        print(f"Number of channels with significant IC enhancement according to empirical p-value via permutation null distribution: {len(significant_channels)} / {len(valid_eeg_names)}")
        
        if len(significant_channels) > 0:
            print("\nSignificant Channels (FDR Corrected):")
            for ch_idx in significant_channels:
                print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr[ch_idx]:.4f}) Corrected from the null mean enhancement Δz = {corrected_delta_z[ch_idx]:.4f} ")
                

    #old statistical way : without permutations 
    p_values = np.array(p_values_no_perm)
    reject_null, p_values_fdr = mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')
    significant_channels = np.where(reject_null)[0]
        
    #print(f"\nTotal Subjects Processed: {eeg_group_delta_z.shape[0]}")
    #print(f"Overall Mean Scalp Enhancement: {np.mean(mean_delta_z):.4f}")
    print(f"\n\nNumber of channels with significant IC enhancement according to jackknife testing: {len(significant_channels)} / {len(valid_eeg_names)}")
        
    if len(significant_channels) > 0:
        print("\nSignificant Channels (FDR Corrected):")
        for ch_idx in significant_channels:
            print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr[ch_idx]:.4f})")
    ####
    

    # ====================================================================
    # Computing Global-Averaged Statistics (All Channels)
    # ====================================================================
    print("\n\n==========================================")
    print(" Computing Global-Averaged Statistics...")
    print("==========================================")

    # Average across ALL channels for each subject
    subj_global_averages = np.mean(eeg_group_delta_z, axis=1) 
    mean_global_enhancement = np.mean(subj_global_averages) 

    print(f"Global Channels Used: All {len(valid_eeg_names)} electrodes")
    print(f"Raw Mean Global Enhancement: {mean_global_enhancement:.4f}")

    # permutation statistical testing
    if n_perms:
        print("  -> Using Permutation Testing for Global Average...")
        
        # Average across all channels for every permutation
        # group_null_delta_z_dist shape: (n_perms, n_channels) -> (n_perms,)
        global_null_averages = np.mean(group_null_delta_z_dist, axis=1)
        
        # Calculate the noise floor and the corrected enhancement
        mean_global_null = np.mean(global_null_averages)
        corrected_global_enhancement = mean_global_enhancement - mean_global_null
        
        # Calculate p-value
        p_val_global = np.sum(global_null_averages >= mean_global_enhancement) / n_perms
        
        print(f"Corrected Mean Global Enhancement: {corrected_global_enhancement:.4f}")
        if p_val_global < 0.05:
            print(f"Global Permutation Test: p = {p_val_global:.4f} *** SIGNIFICANT ***")
        else:
            print(f"Global Permutation Test: p = {p_val_global:.4f} (Not Significant)")

    #fallback to standard parametric/non-parametric tests
    stat_shap_glob, p_shapiro_glob = shapiro(subj_global_averages)
        
    if p_shapiro_glob < 0.05:
        print("  -> Normality violated (Shapiro-Wilk p < 0.05). Using Non-Parametric Wilcoxon Test.")
        res_glob = wilcoxon(subj_global_averages, alternative='greater')
        p_val_glob_fallback = res_glob.pvalue
        test_name_glob = "Wilcoxon Signed-Rank"
    else:
        print("  -> Data is normally distributed. Using 1-Sample T-Test.")
        res_glob = ttest_1samp(subj_global_averages, popmean=0.0, alternative='greater')
        p_val_glob_fallback = res_glob.pvalue
        test_name_glob = "1-Sample T-Test"

    if p_val_glob_fallback < 0.05:
        print(f"Global {test_name_glob}: p = {p_val_glob_fallback:.4f} *** SIGNIFICANT ***")
    else:
        print(f"Global {test_name_glob}: p = {p_val_glob_fallback:.4f} (Not Significant)")


    # ====================================================================
    # Computing ROI-Averaged Statistics
    # ====================================================================
    print("\n==========================================")
    print(" Computing ROI-Averaged Statistics...")
    print("==========================================")

    roi_names = ['Fz', 'Cz', 'FC1', 'FC2', 'C3', 'C4', 'FC5', 'FC6', 'T7', 'T8', 'CP5', 'CP6'] 
    roi_indices = [valid_eeg_names.index(ch) for ch in roi_names if ch in valid_eeg_names]

    if roi_indices:
        # Calculating true ROI enhancement
        roi_delta_z = eeg_group_delta_z[:, roi_indices]
        subj_roi_averages = np.mean(roi_delta_z, axis=1) # Average across channels for each subject
        mean_roi_enhancement = np.mean(subj_roi_averages) # Group average ROI enhancement

        print(f"ROI Channels Used: {[valid_eeg_names[i] for i in roi_indices]}")
        print(f"Raw Mean ROI Enhancement: {mean_roi_enhancement:.4f}")

        # permutation statistical testing
        if n_perms:
            print("  -> Using Permutation Testing for ROI...")
            
            # extracting the ROI channels from our pre-computed group null distribution
            # group_null_delta_z_dist shape: (n_perms, n_channels)
            roi_null_dist = group_null_delta_z_dist[:, roi_indices]
            
            # averaging across the ROI channels for every permutation
            # Resulting shape: (n_perms,)
            roi_null_averages = np.mean(roi_null_dist, axis=1)
            
            # Calculate the noise floor and the corrected enhancement
            mean_roi_null = np.mean(roi_null_averages)
            corrected_roi_enhancement = mean_roi_enhancement - mean_roi_null
            
            # Calculate p-value: % of random shuffles that beat the true ROI mean
            p_val_roi = np.sum(roi_null_averages >= mean_roi_enhancement) / n_perms
            
            print(f"Corrected Mean ROI Enhancement: {corrected_roi_enhancement:.4f}")
            if p_val_roi < 0.05:
                print(f"ROI Permutation Test: p = {p_val_roi:.4f} *** SIGNIFICANT ***")
            else:
                print(f"ROI Permutation Test: p = {p_val_roi:.4f} (Not Significant)")


        #fallback to standard parametric/non-parametric tests
        stat_shap_roi, p_shapiro_roi = shapiro(subj_roi_averages)
            
        if p_shapiro_roi < 0.05:
            print("  -> Normality violated (Shapiro-Wilk p < 0.05). Using Non-Parametric Wilcoxon Test.")
            res_roi = wilcoxon(subj_roi_averages, alternative='greater')
            p_val_roi_fallback = res_roi.pvalue
            test_name_roi = "Wilcoxon Signed-Rank"
        else:
            print("  -> Data is normally distributed. Using 1-Sample T-Test.")
            res_roi = ttest_1samp(subj_roi_averages, popmean=0.0, alternative='greater')
            p_val_roi_fallback = res_roi.pvalue
            test_name_roi = "1-Sample T-Test"

        if p_val_roi_fallback < 0.05:
            print(f"ROI {test_name_roi}: p = {p_val_roi_fallback:.4f} *** SIGNIFICANT ***")
        else:
            print(f"ROI {test_name_roi}: p = {p_val_roi_fallback:.4f} (Not Significant)")

    return {
        "mean_delta_z": mean_delta_z,
        "std_delta_z": std_delta_z,
        "p_values_raw": p_values,
        "p_values_fdr": p_values_fdr,
        "significant_mask": reject_null,
        "channel_names": valid_eeg_names
    }





import sys
class OutputLogger(object):
    """Duplicates stdout to both the console and a specified log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run EEG TRF Analysis on Music Datasets.")

    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['openneuro', 'song'],
        help="Specify which dataset pipeline to run: 'openneuro' or 'song'"
    )

    # OpenNeuro paths
    parser.add_argument('--bids_dir', type=str, default="ds002725", help="Path to OpenNeuro BIDS directory")
    parser.add_argument('--env_dir', type=str, default="openneuro_envelopes", help="Path to envelope npy files of audio stimuli")
    parser.add_argument('--ic_dir', type=str, default="/Users/augustin.bouquillard/Downloads/pa_audioic/ds002775-classical_pa-audioic_ic_0.0-0.3-0.5/noise_0.3_corrected_names", help="Path to IC npy files of audio stimuli")
    
    # SONG dataset paths
    parser.add_argument('--mat_dir', type=str, default="../Giorgia_data/song/1-8Hz/Mastoids", help="Path to SONG .mat files")
    parser.add_argument('--npy_dir', type=str, default="../Giorgia_data/features/64Hz/eel_envelope", help="Path to SONG .npy envelope files")
    parser.add_argument('--csv_dir', type=str, default="/Users/augustin.bouquillard/Downloads/pa_audioic/song_pa-audioic_entropy_mc32_nruns1_0.0-0.3-0.5/noise_0.5", help="Path to SONG ic files")
    #parser.add_argument('--csv_dir', type=str, default="/Users/augustin.bouquillard/Downloads/audioic/song_audioic_ic", help="Path to SONG ic files")
    #parser.add_argument('--csv_dir', type=str, default="/Users/augustin.bouquillard/Downloads/audioic/song_audioic_entropy_mc32", help="Path to SONG ic files")
    
    
    # Time-step parameters
    parser.add_argument('--env_dt', type=float, default=0.015625, help="Time step for the acoustic envelope") #to change to 0.004 for openneuro
    parser.add_argument('--ic_dt', type=float, default=0.0928798, help="Time step for the Information Content") #to change to 0.7 for my old IC curves from this summer

    # Modifiers
    parser.add_argument('--banded', action='store_true', help="Use Banded Ridge Regularization for the Full Model")
    
    # --- Permutations ---
    parser.add_argument('--use_perms', action='store_true', help="Trigger null distribution testing via permutations")
    parser.add_argument('--n_perms', type=int, default=100, help="Number of random permutations for Null Distribution")

    args = parser.parse_args()

    # ==========================================
    # BRANCH 1: OpenNeuro ds002725
    # ==========================================
    if args.dataset == 'openneuro':
        outdir="results_openneuro_"+args.ic_dir.split("/")[-1]
        os.makedirs(outdir, exist_ok=True)
        
        # Initialize output logger
        sys.stdout = OutputLogger(os.path.join(outdir, "terminal_output.txt"))
        print("\n=== Launching OpenNeuro Pipeline ===")
        
        results = run_openneuro_dataset_analysis(
            bids_root_dir=args.bids_dir, 
            env_dir=args.env_dir, 
            ic_dir=args.ic_dir,
            env_dt=args.env_dt,
            ic_dt=args.ic_dt,
            use_banded=args.banded,
            outdir=outdir
        )
        
        fig = plot_group_level_topomap(
            mean_scores=results["mean_delta_z"],
            ch_names=results["channel_names"],
            significant_mask=results["significant_mask"],
            score_type="Mean Δz",
            cmap="Reds",
            vmin=0.0
        )
        fig.savefig(outdir+"/group_level_enhancement_openneuro.png", dpi=300)
        print(f"Saved OpenNeuro topomap to {outdir}/group_level_enhancement_openneuro.png")

        raw_p_vals = results["p_values_raw"]
        ch_names = results["channel_names"]
        mean_dz = results["mean_delta_z"]

        print("\n--- Uncorrected (Raw) EEG Enhancements IC vs envelope baseline (p < 0.05) ---")
        found_any = False
        for i, p in enumerate(raw_p_vals):
            if p < 0.05:
                print(f" - {ch_names[i]}: Δz = {mean_dz[i]:.4f} (raw p = {p:.4f})")
                found_any = True
                
        if not found_any:
            print(" No channels reached uncorrected significance.")

    # ==========================================
    # BRANCH 2: SONG Dataset
    # ==========================================
    elif args.dataset == 'song':
        outdir="tmax=0.7_n_perms="+str(args.n_perms)+"results_song_"+args.csv_dir.split("/")[-2]+"_"+args.csv_dir.split("/")[-1]
        os.makedirs(outdir, exist_ok=True)
        
        # Initialize output logger
        sys.stdout = OutputLogger(os.path.join(outdir, "terminal_output.txt"))
        print("\n=== Launching SONG Dataset Pipeline ===")
        
        # Pass n_perms to function conditionally based on arg flag
        n_perms_val = args.n_perms if args.use_perms else None

        results = run_song_dataset_analysis(
            mat_dir=args.mat_dir, 
            env_dir=args.npy_dir,
            csv_dir=args.csv_dir,
            eeg_fs=64,
            env_dt=args.env_dt,  
            ic_dt=args.ic_dt,
            use_banded=args.banded,
            n_perms=n_perms_val,
            outdir=outdir
        )
        
        fig = plot_group_level_topomap(
            mean_scores=results["mean_delta_z"],
            ch_names=results["channel_names"],
            significant_mask=results["significant_mask"],
            score_type="Mean Δz",
            cmap="Reds",
            vmin=0.0 
        )
        fig.savefig(outdir+"/group_level_enhancement_song.png", dpi=300)
        print(f"Saved SONG topomap to {outdir}/group_level_enhancement_song.png")

        raw_p_vals = results["p_values_raw"]
        ch_names = results["channel_names"]
        mean_dz = results["mean_delta_z"]

        print("\n--- Uncorrected Enhancements (p < 0.05) ---")
        found_any = False
        for i, p in enumerate(raw_p_vals):
            if p < 0.05:
                # Clarify label depending on if perms were used
                stat_type = "perm p" if args.use_perms else "raw p"
                print(f" - {ch_names[i]}: Δz = {mean_dz[i]:.4f} ({stat_type} = {p:.4f})")
                found_any = True
                
        if not found_any:
            print(" No channels reached uncorrected significance.")
