import numpy as np
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval

#from eeg_utils.mtrf.matrices import banded_regularization

#from eeg_utils.mtrf.stats import *

from eeg_utils.utils.utils_statistics import bootstrap_permutation_test, jackknife_corr

from eeg_utils.utils.fastcorr import *

#NOTES MEETING 10/04
#TO DO : compare the results with those obtained on SONG dataset (1-8 Hz) to check my implementation, and the Bach dataset too
#TO DO (DONE) : implement more preprocessing on eeg, especially for openneuro dataset which is pretty raw ; channel rejection, substracting eye component in the EEG (check if there is an EOG channel) data and then filtering ; the filtering is the most important part, 1-8hz (theta and delta bands) is mostly what I am interested in.
#TO DO : downsample the EEG to 32Hz, Mathias apparently didn't have to interpolate IC when doing the analysis (ask him he did it)!
#TO DO : currently there are delay boundary effects, regularize within the training set and remove the average level difference across the trials using the bias terms given by the trf ridge regression
#TO DO : downsampling and SSP could be used to eliminate the stationary noise from the MRI machine DONE (only the first component currently eliminated) 
#TO DO (DONE) : check if the envelope is itself significant to predict EEG, maybe think of other features to check 
#NB: Giorgia didn't use banded_regularization
#TO DO : entropy is at the heart of a lot of neuroscience research - do the analyses on the entropy too (entropy in the predictive coding framework is simply the inverse of precision)
#TO DO : extend the task to the available behavioural data (valence and arousal) from openneuro ds002725 dataset

def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """Applies Fisher z-transformation to Pearson's r values to normalize variance."""
    r_clipped = np.clip(r, -0.999, 0.999)
    return np.arctanh(r_clipped)

def compare_direct_models(stimuli_list, eeg_list, fs):
    """
    Method 1: Comparison between baseline (energy only) and full (energy + IC) 
    using the official mTRFpy Ridge Regression.
    """
    tmin, tmax = -0.1, 1.0 #0.7 to do like in the paper
    k_folds = -1  # Leave-One-Out CV
    regularization = np.logspace(-1, 4, 6)
    trf = TRF(direction=1)

    # Baseline model (Energy only)
    baseline_stimuli = [stim[:, 0:1] for stim in stimuli_list]
    print("Fitting Baseline Model (Energy only)...") 
    r_base, best_lambdas_base = nested_crossval(trf, baseline_stimuli, eeg_list, fs, tmin, tmax, regularization, k=k_folds, average=True, verbose=False)
    opt_lambda_env = np.median(best_lambdas_base)
    
    r_base = crossval(trf, baseline_stimuli, eeg_list, fs, tmin, tmax, opt_lambda_env, k=k_folds, average=False, verbose=False)
    if r_base.ndim == 2:
        r_base = np.mean(r_base, axis=0)
    z_base = fisher_z_transform(r_base)
    print(f'Best energy-only regularization parameter: {opt_lambda_env}')


    # Full model (Energy + IC)
    print("Fitting Full Model (Energy + IC)...")
    # Because both predictors are Z-scored, standard Ridge without banded regularization should be okay 
    r_full, best_lambdas_full = nested_crossval(trf, stimuli_list, eeg_list, fs, tmin, tmax, regularization, k=k_folds, average=True, verbose=False)
    opt_lambda_full = np.median(best_lambdas_full)

    r_full = crossval(trf, stimuli_list, eeg_list, fs, tmin, tmax, opt_lambda_full, k=k_folds, average=False, verbose=False)
    
    if r_full.ndim == 2:
        r_full = np.mean(r_full, axis=0)
    z_full = fisher_z_transform(r_full)

    # Predictive enhancement
    delta_z = z_full - z_base

    return z_full, z_base, delta_z, opt_lambda_full


def evaluate_with_circular_shift(stimuli_list, eeg_list, fs, lambda_val=1.0, n_perms=100, min_shift_sec=10.0, nested=True):#actually linear shifting would be even better but costs us in terms of data quantity
    """
    Method 2: Full model vs. a Null distribution created via circular shifting.

    Parameters:
    - n_perms: number of random shifts to perform.
    - min_shift_sec: minimum temporal shift in seconds to ensure alignment is broken.
    """
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


def evaluate_with_mismatched_epochs(stimuli_list, eeg_list, fs, opt_lambda_full, n_perms=100): #BEST OPTION
    """
    State-of-the-Art Null Distribution: 
    Predicts EEG using the acoustic/IC features from the WRONG epochs.
    Perfectly preserves autocorrelation while destroying mutual information.
    """
    tmin, tmax = -0.1, 0.7
    k_folds = -1
    n_channels = eeg_list[0].shape[1]
    n_epochs = len(stimuli_list)
    
    if n_epochs < 3:
        raise ValueError("Not enough epochs to perform rigorous mismatching.")

    trf = TRF(direction=1)
    
    # evaluating true model using the banded lambdas
    print("Fitting True Model for Permutation Testing...")
    r_full = crossval(
        trf, stimuli_list, eeg_list, fs, tmin, tmax, 
        regularization=opt_lambda_full, bands=[1, 1], method='banded', 
        k=k_folds, average=False, verbose=False
    )
    if r_full.ndim == 2:
        r_full = np.mean(r_full, axis=0)
    z_full = fisher_z_transform(r_full)

    # null distribution via mismatching
    print(f"Generating Null Distribution with {n_perms} Mismatched Permutations...")
    z_null_permutations = np.zeros((n_perms, n_channels))

    for p in range(n_perms):
        # creating a shuffled index list ensuring no epoch matches its original index (Derangement)
        shuffled_indices = np.arange(n_epochs)
        while np.any(shuffled_indices == np.arange(n_epochs)):
            np.random.shuffle(shuffled_indices)
            
        # reordering the stimuli list based on the mismatched indices
        mismatched_stimuli = [stimuli_list[i] for i in shuffled_indices]

        # fitting the model on the completely mismatched data
        r_null = crossval(
            trf, mismatched_stimuli, eeg_list, fs, tmin, tmax, 
            regularization=opt_lambda_full, bands=[1, 1], method='banded', 
            k=k_folds, average=False, verbose=False
        )
        
        if r_null.ndim == 2:
            r_null = np.mean(r_null, axis=0)
            
        z_null_permutations[p, :] = fisher_z_transform(r_null)

    # calculating statistics
    z_null_mean = np.mean(z_null_permutations, axis=0)
    delta_z_null = z_full - z_null_mean

    # empirical p-value: what % of random mismatches accidentally beat the true model?
    p_values = np.sum(z_null_permutations >= z_full, axis=0) / n_perms

    return z_full, z_null_mean, delta_z_null, p_values

#I could cross validate correlations between EEG and stimulus waveform to find the best hardware delay, but there may be no need with TRF
"""
# Example Usage
if __name__ == "__main__":
    #simulating 18 trials (songs) of 2 minutes each at 64 Hz
    n_trials = 18
    fs = 64
    n_samples = 120 * fs

    print("Generating dummy data...")
    dummy_stimuli = [np.random.randn(n_samples, 2) for _ in range(n_trials)]
    dummy_eeg = [np.random.randn(n_samples, 64) for _ in range(n_trials)]

    # --- Method 1 ---
    print("\n--- Running Direct Comparison (E vs E+IC) ---")
    z_f, z_b, delta_z = compare_direct_models(dummy_stimuli, dummy_eeg, fs)
    print(f"Mean Enhancement (Delta Z): {delta_z.mean():.4f}")

    # --- Method 2 ---
    print("\n--- Running Circular Shift Permutations ---")
    # Setting n_perms=5 here just so the example runs fast, use 100 for real data
    z_f, z_null, delta_z_null, p_vals = evaluate_with_circular_shift(dummy_stimuli, dummy_eeg, fs, n_perms=5)
    print(f"Mean Enhancement over Null: {delta_z_null.mean():.4f}")
    print(f"Number of channels with p < 0.05: {np.sum(p_vals < 0.05)} / 64")
"""
##########################################################################################################################################
#Plotting topomap functions
##########################################################################################################################################
import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ttest_1samp, wilcoxon
import mne
import matplotlib.pyplot as plt
from mtrf.model import TRF


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



#######################################################################################################################
#Preprocessing and eeg preparation and alignment
#######################################################################################################################
#quality evaluation :

def plot_ica_diagnostics(raw_before, ica, subj_id):
    raw_eeg_only = raw_before.copy().pick(picks='eeg')
    
    diag_dir = f"results/diagnostics/{subj_id}"
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
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy.stats import zscore

def apply_bandpass_filter(raw: mne.io.BaseRaw, l_freq: float = 1.0, h_freq: float = 8.0) -> mne.io.BaseRaw:
    print(f"    -> Bandpass filtering ({l_freq} - {h_freq} Hz)...")
    # strictly picking EEG only so we don't accidentally filter and destroy the 'music' trigger channel
    picks = mne.pick_types(raw.info, eeg=True)
    raw.filter(l_freq, h_freq, picks=picks, fir_design='firwin', phase='zero', verbose=False)
    return raw
"""
def reject_bad_channels(raw: mne.io.BaseRaw, threshold_uv: float = 500.0) -> mne.io.BaseRaw:
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

def remove_MRI_stationary_noise_with_SSP(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Computes Signal Space Projection (SSP) to remove the highest-variance 
    stationary background noise (often continuous MRI machine hum).
    """
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
        
    return raw

def standardize_trial(eeg_matrix: np.ndarray) -> np.ndarray: #!this is currently done in the main analysis loop!
    """
    Step 4: Z-scores the EEG data across time to remove trial-to-trial DC offsets 
    and scale variance, preventing TRF bias.
    """
    print("    -> Z-scoring trial to remove baseline offsets...")
    # axis=0 standardizes across the time dimension for each channel independently
    return zscore(eeg_matrix, axis=0) 

def clean_eeg_mne_object(raw, subj_id):
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
        
    clean_raw.set_channel_types(mapping)

    #montage
    montage = mne.channels.make_standard_montage('standard_1005')
    clean_raw.set_montage(montage, match_case=False, on_missing='ignore')
    
    #downsampling to 64 Hz before filtering and ICA saves memory and time
    print("    -> Downsampling EEG to 64 Hz...")
    clean_raw.resample(64.0, npad='auto', verbose=False)

    #filtering
    clean_raw.filter(l_freq=1.0, h_freq=8.0, fir_design='firwin', verbose=False)

    clean_raw = find_bad_channels_by_correlation(clean_raw, corr_thresh=0.4)
    if clean_raw.info['bads']:
        clean_raw.interpolate_bads(reset_bads=True, verbose=False)

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

    #diagnostic plots 
    plot_ica_diagnostics(clean_raw, ica, subj_id)

    ica.apply(clean_raw, verbose=False)
    
    return clean_raw
    


############################################################################
#specific to openneuro ds002725 :

def decode_music_channel(voltage_value,feature_dir):
    """
    Decodes the voltage from the 'music' channel into the stimuli filename.
    """
        
    number = int(round(voltage_value * 10**6 * 20))
    str_num = str(number)
    
    for f in os.listdir(feature_dir):
        if "p"+str_num in f :
            print(f)
            return f
    else:
        print("Unknown track"+str_num)

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
        
        # maximum duration in seconds to know how many EEG samples to fill
        max_duration_sec = max(env_time[-1], ic_time[-1]) if len(env_time)>0 and len(ic_time)>0 else 0
        n_eeg_samples = int(np.ceil(max_duration_sec * eeg_fs))
        eeg_time = np.arange(n_eeg_samples) / eeg_fs
        
        # Interpolate Envelope (linear is usually best for continuous audio energy)
        env_interpolator = interp1d(
            env_time, env_data, kind='linear', bounds_error=False, fill_value=0.0
        )
        env_aligned = env_interpolator(eeg_time)
        
        # Interpolate IC 
        ic_interpolator = interp1d(
            ic_time, ic_data, kind='previous', bounds_error=False, fill_value=0.0
        ) #TO DO : what is best here between linear and previous?
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


import os
import glob
import numpy as np
from scipy.stats import ttest_1samp, zscore
import mne
import matplotlib.pyplot as plt
from mtrf.model import TRF

def run_group_level_analysis(bids_root_dir, env_dir, ic_dir, env_dt=0.004, ic_dt=0.7):
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
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
    ]

    for subj_path in subject_folders:
        subj_id = os.path.basename(subj_path)
        print(f"\n=============================")
        print(f" Processing {subj_id}...")
        print(f"=============================")
        
        eeg_list = []
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
                clean_raw = clean_eeg_mne_object(raw, subj_id)
                    
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

                #chunking for "data augmentation" 
                valid_indices = np.where(aligned_stim[:, 0] > 0)[0]
                if len(valid_indices) == 0:
                    continue
                    
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
                                
                                
                                #if np.any(np.std(raw_chunk_stim, axis=0) == 0):
                                #    continue
                                #checking variance only on EEG 
                                if np.any(np.std(raw_chunk_eeg, axis=0) == 0):
                                    continue
                                
                                #standardize and store
                                stimuli_list.append(zscore(raw_chunk_stim, axis=0))
                                eeg_list.append(zscore(raw_chunk_eeg, axis=0))
                                
            except ValueError as e:
                print(f"  [!] {e} Skipping EDF.")
                continue
            
        # ====================================================================
        # Subject-Level TRF Pipeline
        if not stimuli_list:
            continue
            
        try:
            print(f"  -> Fitting direct models (Cross-Validation on {len(stimuli_list)} epochs)...")

            z_full, z_base, delta_z, opt_lambda_full = compare_direct_models(stimuli_list, eeg_list, eeg_fs)
            
            delta_z = np.nan_to_num(delta_z, nan=0.0)
            z_base = np.nan_to_num(z_base, nan=0.0)
            all_delta_z.append(delta_z)
            all_z_base.append(z_base)
            print(f"[{subj_id}] Mean channel enhancement: {np.mean(delta_z):.4f}")

            #plotting topomap according to the value of predictive enhancement obtained from IC in each channel
            eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
            valid_eeg_names = [channel_names[i] for i in eeg_indices]
            scalp_scores = delta_z[eeg_indices]

            os.makedirs("results", exist_ok=True)

            fig = plot_static_score_topomap(
                scores=scalp_scores, 
                ch_names=valid_eeg_names, 
                subj_id=subj_id,
                score_type="Δz (Full - Baseline)",
                cmap="Reds",
                vmin=0.0  
            )
            fig.savefig(f"results/topomap_{subj_id}_delta_z.png", dpi=300)
            plt.close(fig) 

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
            
            #fitting one final full model for each subject on all trials to get the weights for each lag 
            print(f"  -> Fitting final full model to extract weights")
            all_stimuli_concat = np.vstack(stimuli_list)
            all_eeg_concat = np.vstack(eeg_list)

            final_trf = TRF(direction=1)

            final_trf.train(
                all_stimuli_concat, 
                all_eeg_concat, 
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
            fig.savefig(f"results/topomap_{subj_id}_IC_weights.png", dpi=300)
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
    
    #baseline predictions z-transformed pearson correlation coefficients 
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
    print(f"Number of channels with significant enhancement: {len(significant_channels)} / {len(valid_eeg_names)}")
    
    if len(significant_channels) > 0:
        print("\nSignificant Channels (FDR Corrected):")
        for ch_idx in significant_channels:
            print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr[ch_idx]:.4f})")


    return {
        "mean_delta_z": mean_delta_z,
        "std_delta_z": std_delta_z,
        "p_values_raw": p_values,
        "p_values_fdr": p_values_fdr,
        "significant_mask": reject_null,
        "channel_names": valid_eeg_names
    }


#################################################################################################################################################################
#SONG dataset treatment

import scipy.io as sio
import pandas as pd

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import zscore

def prepare_song_trial(eeg_matrix, env_path, ic_csv_path, eeg_fs, env_dt=0.015625, ic_dt=None):
    """
    Takes the pre-cut EEG matrix, loads the standalone Envelope (.npy) and IC (.csv), 
    and perfectly aligns everything to the EEG sampling rate.
    """
    # Information Content from the CSV properly
    ic_df = pd.read_csv(ic_csv_path)
    ic_time = ic_df['time'].values
    

    # We use .fillna(0.0) to turn the empty spaces between notes into zeros.
    # for testing Entropy instead, just need to change 'nll_0.000' to 'entr_0.000'
    ic_data = ic_df['nll_0.000'].fillna(0.0).values
    
    #acoustic envelope
    env_data = np.squeeze(np.load(env_path))
    #time vectors
    eeg_length = eeg_matrix.shape[0]
    eeg_time = np.arange(eeg_length) / eeg_fs
    
    env_time = np.arange(len(env_data)) * env_dt
    
    #Interpolate IC using the CSV's explicit time vector
    ic_interpolator = interp1d(ic_time, ic_data, kind='linear', bounds_error=False, fill_value=0.0)
    ic_aligned = ic_interpolator(eeg_time)
    
    #Interpolate Envelope
    env_interpolator = interp1d(env_time, env_data, kind='linear', bounds_error=False, fill_value=0.0)
    env_aligned = env_interpolator(eeg_time)
    
    #Combine Acoustic Envelope and IC into one Stimulus Matrix
    aligned_stim = np.column_stack((env_aligned, ic_aligned))
    
    #Z-score both the Stimulus and the EEG to remove baseline biases
    clean_stim = np.nan_to_num(zscore(aligned_stim, axis=0))
    clean_eeg = np.nan_to_num(zscore(eeg_matrix, axis=0))
    
    return clean_stim, clean_eeg

def run_song_dataset_analysis(mat_dir, env_dir, csv_dir, eeg_fs=None, env_dt=0.004, ic_dt=0.5):
    
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "pre_dataSub*.mat")))
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "ic_*_song.csv")))
    
    all_delta_z = []
    all_z_base = []
    
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
        
        # 1. Load the preprocessed subject data
        mat_data = sio.loadmat(mat_file)
        eeg_struct = mat_data['eeg'][0, 0]
        
        trial_fs = int(eeg_struct['fs'].item())
        if eeg_fs is None:
            eeg_fs = trial_fs
            
        eeg_trials = eeg_struct['data'][0]
        
        # extracting channel names from the MATLAB struct on the first pass!
        if channel_names is None:
            # chanlocs is usually a structured array where 'labels' holds the name
            chanlocs = eeg_struct['chanlocs'][0]
            channel_names = [ch['labels'][0] for ch in chanlocs]
            print(f"    -> Extracted {len(channel_names)} channels from MAT file.")
        
        eeg_list = []
        stimuli_list = []
        
        # TO DO: generalise if needed - for now this strictly assumes eeg_trials[0] is ic_01_song, eeg_trials[1] is ic_02_song, etc.
        for trial_idx in range(len(csv_files)):
            
            # Extract the raw EEG array from the MATLAB cell
            trial_eeg = eeg_trials[trial_idx] 
            
            # Safely catch and skip empty MATLAB arrays (e.g., rejected trials)
            if trial_eeg.size == 0 or max(trial_eeg.shape) < (eeg_fs * 2):
                print(f"  [!] Warning: Trial {trial_idx} is empty or too short (< 2s). Skipping.")
                continue
                
            # Squeeze out any hidden MATLAB cell dimensions
            trial_eeg = np.squeeze(trial_eeg)
            if len(trial_eeg.shape) < 2:
                print(f"  [!] Warning: Trial {trial_idx} has invalid dimensions. Skipping.")
                continue
                
            # Transpose to (time, channels) if necessary
            if trial_eeg.shape[0] < trial_eeg.shape[1]:
                trial_eeg = trial_eeg.T
                
            current_csv_path = csv_files[trial_idx]
            
            # --- THE FILE MATCHING MAGIC ---
            csv_filename = os.path.basename(current_csv_path)
            env_filename = csv_filename.replace('ic_', '').replace('.csv', '.npy')
            env_path = os.path.join(env_dir, env_filename)
            
            if not os.path.exists(env_path):
                print(f"  [!] Warning: Envelope {env_filename} not found. Skipping trial.")
                continue
            # --------------------------------
            
            # Prepare and align
            clean_stim, clean_eeg = prepare_song_trial(
                eeg_matrix=trial_eeg, 
                env_path=env_path, 
                ic_csv_path=current_csv_path, 
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
        z_full, z_base, delta_z, opt_lam = compare_direct_models(stimuli_list, eeg_list, eeg_fs)
        
        delta_z = np.nan_to_num(delta_z, nan=0.0)
        z_base = np.nan_to_num(z_base, nan=0.0) # Ensure z_base is also cleaned
        
        all_delta_z.append(delta_z)
        all_z_base.append(z_base) # Ensure we track this for the group stats
        print(f"[{subj_id}] Mean channel enhancement: {np.mean(delta_z):.4f}")
        
        
    # ====================================================================
    # Group-Level Statistics 

    print("\n==========================================")
    print(" Computing Group-Level Statistics...")
    print("==========================================")

    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]
    
    #baseline predictions z-transformed pearson correlation coefficients 
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
    print(f"Number of channels with significant enhancement: {len(significant_channels)} / {len(valid_eeg_names)}")
    
    if len(significant_channels) > 0:
        print("\nSignificant Channels (FDR Corrected):")
        for ch_idx in significant_channels:
            print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr[ch_idx]:.4f})")


    return {
        "mean_delta_z": mean_delta_z,
        "std_delta_z": std_delta_z,
        "p_values_raw": p_values,
        "p_values_fdr": p_values_fdr,
        "significant_mask": reject_null,
        "channel_names": valid_eeg_names
    }


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

    #path arguments (with defaults)
    # OpenNeuro paths
    parser.add_argument('--bids_dir', type=str, default="ds002725", help="Path to OpenNeuro BIDS directory")
    parser.add_argument('--env_dir', type=str, default="enveloppe_of_stimuli", help="Path to envelope npy files of audio stimuli")
    parser.add_argument('--ic_dir', type=str, default="ic_curves_global_stimuli", help="Path to IC npy files of audio stimuli")
    
    # SONG dataset paths
    parser.add_argument('--mat_dir', type=str, default="../Giorgia_data/song/1-8Hz/Mastoids", help="Path to SONG .mat files")
    parser.add_argument('--csv_dir', type=str, default="../Giorgia_data/features/ic_neurips/aligned", help="Path to SONG .csv ic files")
    parser.add_argument('--npy_dir', type=str, default="../Giorgia_data/features/64Hz/eel_envelope", help="Path to SONG .npy envelope files")

    #parsing arguments from the terminal
    args = parser.parse_args()

    #creating results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # ==========================================
    # BRANCH 1: OpenNeuro ds002725
    # ==========================================
    if args.dataset == 'openneuro':
        print("\n=== Launching OpenNeuro Pipeline ===")
        results = run_group_level_analysis(
            bids_root_dir=args.bids_dir, 
            env_dir=args.env_dir, 
            ic_dir=args.ic_dir
        )
        
        fig = plot_group_level_topomap(
            mean_scores=results["mean_delta_z"],
            ch_names=results["channel_names"],
            significant_mask=results["significant_mask"],
            score_type="Mean Δz",
            cmap="Reds",
            vmin=0.0
        )
        fig.savefig("results/group_level_enhancement_openneuro.png", dpi=300)
        print("Saved OpenNeuro topomap to results/group_level_enhancement_openneuro.png")

    # ==========================================
    # BRANCH 2: SONG Dataset
    # ==========================================
    elif args.dataset == 'song':
        print("\n=== Launching SONG Dataset Pipeline ===")
        
        results = run_song_dataset_analysis(
            mat_dir=args.mat_dir, 
            env_dir=args.npy_dir,
            csv_dir=args.csv_dir,
            eeg_fs=64 
        )
        
        fig = plot_group_level_topomap(
            mean_scores=results["mean_delta_z"],
            ch_names=results["channel_names"],
            significant_mask=results["significant_mask"],
            score_type="Mean Δz",
            cmap="Reds",
            vmin=0.0
        )
        fig.savefig("results/group_level_enhancement_song.png", dpi=300)
        print("Saved SONG topomap to results/group_level_enhancement_song.png")
    
