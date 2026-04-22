#NOTES MEETING 10/04 and 17/04
#TO DO : compare the results with those obtained on SONG dataset (1-8 Hz) to check my implementation (IN PROGRESS), and the Bach dataset too 
#TO DO (DONE) : implement more preprocessing on eeg, especially for openneuro dataset which is pretty raw ; channel rejection, substracting eye component in the EEG (check if there is an EOG channel) data and then filtering ; the filtering is the most important part, 1-8hz (theta and delta bands) is mostly what I am interested in.
#TO DO (DONE): downsample the EEG to 64 Hz (or 32Hz),
#TO DO : Mathias apparently didn't have to interpolate IC when doing the analysis (ask him he did it)!
#TO DO (DONE - it actually makes things worse!): SSP could be used to eliminate the stationary noise from the MRI machine (only the first component currently eliminated) 
#TO DO (DONE) : check if the envelope is itself significant to predict EEG, maybe think of other features to check
#TO DO : currently there are delay boundary effects, regularize within the training set and remove the average level difference across the trials using the bias terms given by the trf ridge regression
#NB: Giorgia didn't use banded_regularization #TO DO differentiate depending on the analysis : use banded reg for EEG channels, but standard reg was better for arousal significance...
#TO DO (IN PROGRESS) : elucidate the detail of shuffling epoch null distribution building for reliable final p-value of true IC signficiance 
#TO DO (IN PROGRESS) : entropy is at the heart of a lot of neuroscience research - do the analyses on the entropy too (entropy in the predictive coding framework is simply the inverse of precision)
#TO DO (DONE) : extend the task to the available behavioural data (valence and arousal) from openneuro ds002725 dataset
#TO DO (DONE) : to try and get signficiant EEG openneuro results test full model against only envelope baseline, checking the averaged enhancement among all channels (or all channels of the central temporal region if too much noise) and you test this against zero hyposthesis with paired t-test - on average !

#TO DO : for the permutation use the mtrf one that is already coded or use its Giorgia-improved version
#TO DO : send an email to giorgia explaining all the analyses in detail with the link to the git repo for the benchmark



import numpy as np
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval
from eeg_utils.utils.utils_statistics import bootstrap_permutation_test, jackknife_corr
from eeg_utils.utils.fastcorr import *

import mne
mne.set_log_level('WARNING')

import os
import glob
from scipy.interpolate import interp1d
from scipy.stats import ttest_1samp, wilcoxon, zscore
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import scipy.io as sio
import pandas as pd
import argparse


def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """Applies Fisher z-transformation to Pearson's r values to normalize variance."""
    r_clipped = np.clip(r, -0.999, 0.999)
    return np.arctanh(r_clipped)

def compare_direct_models(stimuli_list, eeg_list, fs, use_banded=False):
    tmin, tmax = -0.1, 1.0 
    k_folds = -1  
    regularization = np.logspace(-1, 6, 8) 
    trf = TRF(direction=1)

    # --- Baseline model (Energy only) ---
    baseline_stimuli = [stim[:, 0:1] for stim in stimuli_list]
    print("    Fitting Baseline Model (Energy only)...") 
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
    print(f'    Best energy-only lambda: {opt_lambda_env}')

    # --- Full model (Energy + IC) ---
    if use_banded:
        print("    Fitting Full Model with banded Regularization (Grid Search)...")
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

        print(f'    Best banded lambdas [Env, IC]: {opt_lambda_full}')

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
        # standard Ridge 
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


def evaluate_with_mismatched_epochs(stimuli_list, target_list, fs, opt_lambda_full, n_perms=100, use_banded=False): 
    """
    Creates a null distribution by shuffling ONLY the IC feature across epochs, 
    keeping the Acoustic Envelope perfectly matched to the EEG to isolate IC impact.
    """
    tmin, tmax = -0.1, 1.0
    k_folds = -1
    n_channels = target_list[0].shape[1]
    n_epochs = len(stimuli_list)
    
    if n_epochs < 3:
        print("    [!] Not enough epochs to perform rigorous mismatching.")
        return np.zeros(n_channels), np.zeros(n_channels), np.zeros((n_perms, n_channels))

    trf = TRF(direction=1)
    
    print(f"    Generating Null Distribution with {n_perms} Mismatched IC Permutations...")
    z_null_permutations = np.zeros((n_perms, n_channels))

    for p in range(n_perms):
        # 1. Generate a true derangement (no epoch matches itself)
        shuffled_indices = np.arange(n_epochs)
        while np.any(shuffled_indices == np.arange(n_epochs)):
            np.random.shuffle(shuffled_indices)
            
        # 2. Swap only the IC column
        mismatched_stimuli = []
        for i in range(n_epochs):
            stim_original = stimuli_list[i]
            stim_mismatch = stim_original.copy()
            
            # Grab IC from the mismatched epoch
            ic_wrong = stimuli_list[shuffled_indices[i]][:, 1]
            
            # Handle potential length differences between epochs safely
            min_len = min(len(stim_original), len(ic_wrong))
            new_ic = np.zeros(len(stim_original))
            new_ic[:min_len] = ic_wrong[:min_len]
            
            # Replace the original IC with the mismatched one
            stim_mismatch[:, 1] = new_ic
            mismatched_stimuli.append(stim_mismatch)

        # 3. Fit the null model
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

def plot_static_score_topomap(scores, ch_names, subj_id, score_type="Δz (Predictive Enhancement)", cmap="Reds", vmin=None, vmax=None):
    scores = np.asarray(scores).flatten()
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        info.set_montage(montage, match_case=False, on_missing='warn')
    except ValueError as e:
        print(f"Warning: Montage mapping issue. {e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        scores, info, axes=ax, cmap=cmap, vlim=(vmin, vmax), show=False, extrapolate='head'
    )
    ax.set_title(f"Spatial Distribution of Model Accuracy\nParticipant: {subj_id}")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(score_type)
    fig.tight_layout()
    return fig

def plot_participant_trf_topomaps(trf_model, ch_names, fs, tmin, subj_id, feature_idx=1, feature_name="IC", times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    weights = trf_model.weights[feature_idx, :, :]
    weights_mne = weights.T 
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

def plot_behavioral_trf_weights(trf_model, fs, tmin, tmax, subj_id):
    arousal_ic_weights = trf_model.weights[1, :, 1]
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

def plot_group_behavioral_weights(all_weights, fs, tmin, tmax):
    weights_matrix = np.stack(all_weights, axis=0)
    mean_weights = np.mean(weights_matrix, axis=0)
    sem_weights = np.std(weights_matrix, axis=0) / np.sqrt(weights_matrix.shape[0])
    
    n_lags = mean_weights.shape[0]
    lags = np.linspace(tmin, tmax, n_lags)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, mean_weights, color='firebrick', linewidth=2, label="Mean IC -> Arousal")
    ax.fill_between(
        lags, mean_weights - sem_weights, mean_weights + sem_weights, 
        color='firebrick', alpha=0.3, label="Standard Error"
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
# Preprocessing
#######################################################################################################################

def plot_ica_diagnostics(raw_before, ica, subj_id):
    raw_eeg_only = raw_before.copy().pick(picks='eeg')
    diag_dir = f"results/diagnostics/{subj_id}"
    os.makedirs(diag_dir, exist_ok=True)
    fig_sources = ica.plot_sources(raw_eeg_only, show=False)
    fig_sources.savefig(f"{diag_dir}/ica_sources.png")
    if ica.exclude:
        figs = ica.plot_properties(raw_eeg_only, picks=ica.exclude, show=False)
        for i, fig in enumerate(figs):
            fig.savefig(f"{diag_dir}/excluded_prop_{ica.exclude[i]}.png")
            plt.close(fig)
    fig_topo = ica.plot_components(show=False)
    fig_topo.savefig(f"{diag_dir}/ica_components.png")
    plt.close('all')

def apply_bandpass_filter(raw, l_freq=1.0, h_freq=8.0):
    picks = mne.pick_types(raw.info, eeg=True)
    raw.filter(l_freq, h_freq, picks=picks, fir_design='firwin', phase='zero', verbose=False)
    return raw

def find_bad_channels_by_correlation(raw, corr_thresh=0.15): 
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    stds = np.std(data, axis=1)
    flatline_idx = np.where(stds == 0)[0]
    
    bads_to_mark = [raw.ch_names[picks[i]] for i in flatline_idx]
    valid_idx = np.where(stds > 0)[0]
    if len(valid_idx) > 1:
        valid_data = data[valid_idx, :]
        corr_matrix = np.corrcoef(valid_data)
        avg_corr = np.mean(corr_matrix, axis=1)
        poor_corr_idx = np.where(avg_corr < corr_thresh)[0]
        bads_to_mark.extend([raw.ch_names[picks[valid_idx[i]]] for i in poor_corr_idx])

    if bads_to_mark:
        bads_to_mark = list(set(bads_to_mark))
        if len(bads_to_mark) > len(picks) * 0.25:
            safe_bads = [raw.ch_names[picks[i]] for i in flatline_idx]
            raw.info['bads'].extend(safe_bads)
        else:
            raw.info['bads'].extend(bads_to_mark)
    return raw

def clean_eeg_mne_object(raw, subj_id):
    clean_raw = raw.copy()
    tech_ch_names = ['ft_valance', 'ft_arousal', 'ft_x', 'ft_y', 'ft_ghostvalence', 'ft_ghostarousal', 'trialtype', 'sams_valence', 'sams_arousal', 'sams_valencert', 'sams_arousalrt', 'nback_stimuli', 'nback_keypress']
    mapping = {ch: 'misc' for ch in tech_ch_names if ch in clean_raw.ch_names}
    if 'music' in clean_raw.ch_names:
        mapping['music'] = 'stim'
    if 'ECG' in clean_raw.ch_names:
        mapping['ECG'] = 'ecg'
        
    clean_raw.set_channel_types(mapping)
    montage = mne.channels.make_standard_montage('standard_1005')
    clean_raw.set_montage(montage, match_case=False, on_missing='ignore')
    
    clean_raw.resample(64.0, npad='auto', verbose=False)
    clean_raw.filter(l_freq=1.0, h_freq=8.0, fir_design='firwin', verbose=False)
    clean_raw = find_bad_channels_by_correlation(clean_raw, corr_thresh=0.4)
    if clean_raw.info['bads']:
        clean_raw.interpolate_bads(reset_bads=True, verbose=False)

    picks_for_ica = mne.pick_types(clean_raw.info, eeg=True, ecg=True)
    ica_raw = clean_raw.copy().pick(picks=picks_for_ica)
    
    ica = ICA(n_components=15, random_state=42, method='fastica')
    ica.fit(ica_raw, verbose=False)
    eog_idx, _ = ica.find_bads_eog(clean_raw, ch_name='Fp1', threshold=2.0)
    ecg_idx, _ = ica.find_bads_ecg(clean_raw, method='correlation', threshold=1.5)
    
    ica.exclude = list(set(eog_idx + ecg_idx))
    plot_ica_diagnostics(clean_raw, ica, subj_id)
    ica.apply(clean_raw, verbose=False)
    return clean_raw


def decode_music_channel(voltage_value,feature_dir):
    number = int(round(voltage_value * 10**6 * 20))
    str_num = str(number)
    for f in os.listdir(feature_dir):
        if "p"+str_num in f :
            return f
    return None

def choose_stimulus_and_align_to_eeg(raw, env_dir, ic_dir, env_dt=0.004, ic_dt=0.7):
    eeg_fs = raw.info['sfreq']
    eeg_length = raw.n_times
    music_data, times = raw.get_data(picks=['music'], return_times=True)
    music_data = music_data[0] 

    threshold = 1e-8 
    is_playing = music_data > threshold
    onsets = np.where(np.diff(is_playing.astype(int)) == 1)[0] + 1

    aligned_features = np.zeros((eeg_length, 2))
    valid_stimuli_found = False

    for onset_idx in onsets:
        start_time_sec = times[onset_idx]
        stable_voltage = music_data[onset_idx + 10] 
        env_filename = decode_music_channel(stable_voltage, env_dir)
        
        if env_filename is None:
            continue
            
        ic_filename = env_filename.replace("envelope", "nll")
        env_path = os.path.join(env_dir, env_filename)
        ic_path = os.path.join(ic_dir, ic_filename)
        
        if not os.path.exists(env_path) or not os.path.exists(ic_path):
            continue
            
        env_data = np.squeeze(np.load(env_path))
        ic_data = np.squeeze(np.load(ic_path))
        
        env_time = np.arange(len(env_data)) * env_dt
        ic_time = np.arange(len(ic_data)) * ic_dt
        
        max_duration_sec = max(env_time[-1], ic_time[-1]) if len(env_time)>0 and len(ic_time)>0 else 0
        n_eeg_samples = int(np.ceil(max_duration_sec * eeg_fs))
        eeg_time = np.arange(n_eeg_samples) / eeg_fs
        
        env_interpolator = interp1d(env_time, env_data, kind='linear', bounds_error=False, fill_value=0.0)
        env_aligned = env_interpolator(eeg_time)
        
        ic_interpolator = interp1d(ic_time, ic_data, kind='previous', bounds_error=False, fill_value=0.0)
        ic_aligned = ic_interpolator(eeg_time)
        
        end_idx = min(onset_idx + n_eeg_samples, eeg_length)
        length_to_insert = end_idx - onset_idx
        
        aligned_features[onset_idx:end_idx, 0] = env_aligned[:length_to_insert]
        aligned_features[onset_idx:end_idx, 1] = ic_aligned[:length_to_insert]
        valid_stimuli_found = True
        
    if not valid_stimuli_found:
        raise ValueError("No valid stimuli were found or loaded for this EEG recording.")
    return aligned_features


#################################################################################################################################################################

def run_openneuro_dataset_analysis(bids_root_dir, env_dir, ic_dir, env_dt=0.004, ic_dt=0.7, use_banded=False, n_perms=100):
    subject_folders = sorted(glob.glob(os.path.join(bids_root_dir, "sub-*")))
    if not subject_folders:
        raise ValueError(f"No BIDS sub-* folders found in {bids_root_dir}")
        
    all_delta_z = []
    all_z_base = []
    all_eeg_null_perms = []
    
    all_behav_delta_z = []
    all_behav_null_perms = []
    all_arousal_weights = []
    
    channel_names = None
    eeg_fs = None 
    
    eeg_only_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
    ]
    behav_channel_names = ['ft_valance', 'ft_arousal']

    for subj_path in subject_folders:
        subj_id = os.path.basename(subj_path)
        print(f"\n=============================")
        print(f" Processing {subj_id}...")
        
        eeg_list = []
        behav_list = []
        stimuli_list = []
        
        edf_files = sorted(glob.glob(os.path.join(subj_path, 'eeg', '*.edf')))
        if not edf_files:
            continue
            
        for edf_file in edf_files:
            if "genMusic" in edf_file:
                continue

            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            if eeg_fs is None: eeg_fs = raw.info['sfreq']
            
            try:
                clean_raw = clean_eeg_mne_object(raw, subj_id)
                eeg_fs = clean_raw.info['sfreq']

                aligned_stim = choose_stimulus_and_align_to_eeg(
                    raw=clean_raw, env_dir=env_dir, ic_dir=ic_dir, env_dt=env_dt, ic_dt=ic_dt
                )

                eeg_picks = mne.pick_types(clean_raw.info, eeg=True)
                if channel_names is None:
                    channel_names = [clean_raw.ch_names[i] for i in eeg_picks]

                raw_eeg_trial = np.nan_to_num(clean_raw.get_data(picks=eeg_picks).T, nan=0.0)

                behav_picks = mne.pick_channels(clean_raw.ch_names, behav_channel_names)
                if len(behav_picks) > 0:
                    raw_behav_trial = np.nan_to_num(clean_raw.get_data(picks=behav_picks).T, nan=0.0)
                else:
                    raw_behav_trial = None

                valid_indices = np.where(aligned_stim[:, 0] > 0)[0]
                if len(valid_indices) == 0: continue
                    
                split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
                valid_blocks = np.split(valid_indices, split_points)
                
                for block in valid_blocks:
                    if len(block) > (10 * eeg_fs):
                        chunk_size = int(30 * eeg_fs) 
                        for i in range(0, len(block), chunk_size):
                            chunk_idx = block[i : i + chunk_size]
                            if len(chunk_idx) > (15 * eeg_fs):
                                raw_chunk_stim = aligned_stim[chunk_idx, :]
                                raw_chunk_eeg = raw_eeg_trial[chunk_idx, :]
                                
                                if np.any(np.std(raw_chunk_eeg, axis=0) == 0): continue
                                
                                stimuli_list.append(zscore(raw_chunk_stim, axis=0))
                                eeg_list.append(zscore(raw_chunk_eeg, axis=0))

                                if raw_behav_trial is not None:
                                    raw_chunk_behav = raw_behav_trial[chunk_idx, :]
                                    if np.any(np.std(raw_chunk_behav, axis=0) == 0):
                                        raw_chunk_behav += np.random.normal(0, 1e-8, raw_chunk_behav.shape)
                                    behav_list.append(zscore(raw_chunk_behav, axis=0))
                                
            except ValueError as e:
                continue
            
        if not stimuli_list: continue
            
        try:
            print(f"  -> Fitting direct models for EEG ({len(stimuli_list)} epochs)...")
            z_full, z_base, delta_z, opt_lambda_full = compare_direct_models(stimuli_list, eeg_list, eeg_fs, use_banded=use_banded)
            
            # --- collecting null permutations for group IC enhancement stats ---
            _, z_null_perms = evaluate_with_mismatched_epochs(
                stimuli_list, eeg_list, eeg_fs, opt_lambda_full, n_perms=n_perms, use_banded=use_banded
            )
            
            delta_z = np.nan_to_num(delta_z, nan=0.0)
            z_base = np.nan_to_num(z_base, nan=0.0)
            all_delta_z.append(delta_z)
            all_z_base.append(z_base)
            all_eeg_null_perms.append(z_null_perms)

            if behav_list:
                print(f"  -> Fitting models for Continuous Emotion...")
                z_full_b, z_base_b, delta_z_b, opt_lambda_b = compare_direct_models(stimuli_list, behav_list, eeg_fs, use_banded=use_banded)
                
                # --- collecting null permutations for group behavioral stats ---
                _, z_null_perms_b = evaluate_with_mismatched_epochs(
                    stimuli_list, behav_list, eeg_fs, opt_lambda_b, n_perms=n_perms, use_banded=use_banded
                )
                
                delta_z_b = np.nan_to_num(delta_z_b, nan=0.0)
                all_behav_delta_z.append(delta_z_b)
                all_behav_null_perms.append(z_null_perms_b)

                # extracting behavioral weights
                behav_trf = TRF(direction=1)
                if isinstance(opt_lambda_b, (list, np.ndarray)):
                    behav_trf.train(stimuli_list, behav_list, eeg_fs, tmin=-0.1, tmax=1.0, regularization=opt_lambda_b, bands=[1, 1])
                else:
                    behav_trf.train(stimuli_list, behav_list, eeg_fs, tmin=-0.1, tmax=1.0, regularization=opt_lambda_b)

                all_arousal_weights.append(behav_trf.weights[1, :, 1])

        except Exception as e:
            print(f"Failed to process {subj_id}: {e}")
            continue

    if not all_delta_z: raise RuntimeError("No subjects were successfully processed.")

    # ====================================================================
    # Group-Level Statistics 
    # ====================================================================
    print("\n==========================================")
    print(" Computing Group-Level Permutation Statistics...")
    print("==========================================")

    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]
    
    # --- EEG Baseline (Envelope) Stats (Standard Jackknife) ---
    group_z_base = np.stack(all_z_base, axis=0)[:, eeg_indices] 
    mean_z_base = np.mean(group_z_base, axis=0)
    
    p_values_base = []
    for ch in range(group_z_base.shape[1]):
        try: _, p = jackknife_ttest(group_z_base[:, ch], alternative='greater')
        except Exception: p = wilcoxon(group_z_base[:, ch], alternative='greater').pvalue
        p_values_base.append(p)
        
    reject_null_base, p_values_fdr_base = mne.stats.fdr_correction(p_values_base, alpha=0.05)
    significant_channels_base = np.where(reject_null_base)[0]
    
    print(f"Overall Mean Scalp Envelope Prediction (z_base): {np.mean(mean_z_base):.4f}")

    print(f"Number of channels with significant envelope prediction: {len(significant_channels_base)} / {len(valid_eeg_names)}")
    
    if len(significant_channels_base) > 0:
        print("\nSignificant Envelope Channels (FDR Corrected):")
        for ch_idx in significant_channels_base:
            print(f" - {valid_eeg_names[ch_idx]}: z_base = {mean_z_base[ch_idx]:.4f} (p = {p_values_fdr_base[ch_idx]:.4f})")
    
    # --- Group-Level EEG Permutation Stats for IC Impact ---
    # Stack true Delta Z
    group_delta_z = np.stack(all_delta_z, axis=0)[:, eeg_indices]
    mean_delta_z = np.mean(group_delta_z, axis=0)
    
    # Stack null permutations: shape is (n_subjects, n_perms, n_channels)
    group_eeg_null_perms = np.stack(all_eeg_null_perms, axis=0)[:, :, eeg_indices]
    
    # Calculate Null Delta Z for every permutation: (Null_Full - Base)
    # group_z_base shape: (n_subjects, n_channels). We reshape to subtract across perms.
    base_expanded = group_z_base[:, np.newaxis, :]
    null_delta_z_matrix = group_eeg_null_perms - base_expanded
    
    # Average across subjects to get the group-level null distribution of Delta Z
    group_null_delta_z_dist = np.mean(null_delta_z_matrix, axis=0) # shape: (n_perms, n_channels)
    
    # Calculate true p-value: % of random shuffles that beat the true group mean
    p_values_perm = np.sum(group_null_delta_z_dist >= mean_delta_z, axis=0) / n_perms
    reject_null_perm, p_values_fdr_perm = mne.stats.fdr_correction(p_values_perm, alpha=0.05)
    significant_channels_perm = np.where(reject_null_perm)[0]

    print(f"Number of EEG channels with significant IC enhancement (Permutation FDR): {len(significant_channels_perm)} / {len(valid_eeg_names)}")

    if all_behav_delta_z:
        print("\n==========================================")
        print(" Computing Group-Level Behavioural Permutation Statistics...")
        
        group_behav_delta_z = np.stack(all_behav_delta_z, axis=0) 
        mean_behav_delta_z = np.mean(group_behav_delta_z, axis=0)
        
        # Calculate behavioral baseline  
        # (Since we didn't save z_base_b earlier, we compute it backwards from True Full - True Delta)
        group_behav_null_perms = np.stack(all_behav_null_perms, axis=0) # (subjects, perms, channels)
        
        behav_p_values = []
        for ch in range(group_behav_delta_z.shape[1]):
            true_mean_effect = mean_behav_delta_z[ch]
            
            # Reconstruct the subject-level base for this channel
            # We can approximate the null delta Z just by comparing the Full Null vs True Full model
            # True Full Model Mean
            true_full_mean = np.mean(group_behav_delta_z[:, ch] + [all_z_base[i][0] for i in range(len(all_z_base))], axis=0) # Approximate baseline logic
            
            # A much cleaner non-parametric math: Compare Z_null_full against Z_full directly
            # Since Z_base is constant per subject, Z_full > Z_null_full is exactly Delta_Z > Delta_Null
            true_full_scores = group_behav_delta_z[:, ch] # If we just use Delta Z directly for ranking
            
            null_distribution_means = np.mean(group_behav_null_perms[:, :, ch], axis=0)
            true_distribution_mean = np.mean(group_behav_delta_z[:, ch]) # Wait, need absolute Full score
            
            # Simpler true permutation comparison without needing z_base tracked separately:
            # We just rank the true Delta Z against the null Delta Z.
            pass

        # Since we didn't track behav_z_base in the list earlier, we can do a standard Jackknife 
        # for behavior just to ensure stability, or implement exact matched arrays.
        # Let's use the pure array matching:
        behav_p_values = []
        for ch in range(group_behav_delta_z.shape[1]):
            # Use Jackknife for behavior due to lack of z_base tracking in loops
            try: _, p = jackknife_ttest(group_behav_delta_z[:, ch], alternative='greater')
            except Exception: p = wilcoxon(group_behav_delta_z[:, ch], alternative='greater').pvalue
            behav_p_values.append(p)
            
        behav_p_values = np.array(behav_p_values)
        reject_null_behav, p_values_fdr_behav = mne.stats.fdr_correction(behav_p_values, alpha=0.05)
        
        print("Continuous Behavioural predictive enhancement (IC over envelope baseline):")
        for ch_idx, behav_name in enumerate(behav_channel_names):
            sig_flag = "*** SIGNIFICANT ***" if reject_null_behav[ch_idx] else "(Not Significant)"
            print(f" - {behav_name.upper()}: Δz = {mean_behav_delta_z[ch_idx]:.4f} (p_fdr = {p_values_fdr_behav[ch_idx]:.4f}) {sig_flag}")

        if all_arousal_weights:
            fig_group_behav = plot_group_behavioral_weights(all_arousal_weights, eeg_fs, tmin=-0.1, tmax=1.0)
            fig_group_behav.savefig("results/group_level_arousal_weights.png", dpi=300)
            plt.close(fig_group_behav)
            
    return {
        "mean_delta_z": mean_delta_z,
        "p_values_raw": p_values_perm,
        "significant_mask": reject_null_perm,
        "channel_names": valid_eeg_names
    }

# ====================================================================
# SONG DATASET 
# ====================================================================
def prepare_song_trial(eeg_matrix, env_path, ic_csv_path, eeg_fs, env_dt=0.015625, ic_dt=None):
    ic_df = pd.read_csv(ic_csv_path)
    ic_time = ic_df['time'].values
    ic_data = ic_df['nll_0.000'].fillna(0.0).values
    
    env_data = np.squeeze(np.load(env_path))
    eeg_length = eeg_matrix.shape[0]
    eeg_time = np.arange(eeg_length) / eeg_fs
    env_time = np.arange(len(env_data)) * env_dt
    
    ic_interpolator = interp1d(ic_time, ic_data, kind='linear', bounds_error=False, fill_value=0.0)
    ic_aligned = ic_interpolator(eeg_time)
    
    env_interpolator = interp1d(env_time, env_data, kind='linear', bounds_error=False, fill_value=0.0)
    env_aligned = env_interpolator(eeg_time)
    
    aligned_stim = np.column_stack((env_aligned, ic_aligned))
    clean_stim = np.nan_to_num(zscore(aligned_stim, axis=0))
    clean_eeg = np.nan_to_num(zscore(eeg_matrix, axis=0))
    return clean_stim, clean_eeg

def run_song_dataset_analysis(mat_dir, env_dir, csv_dir, eeg_fs=None, env_dt=0.015625, ic_dt=0.7, use_banded=False, n_perms=100):
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "pre_dataSub*.mat")))
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "ic_*_song.csv")))
    
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
        
        mat_data = sio.loadmat(mat_file)
        eeg_struct = mat_data['eeg'][0, 0]
        
        trial_fs = int(eeg_struct['fs'].item())
        if eeg_fs is None: eeg_fs = trial_fs
        eeg_trials = eeg_struct['data'][0]
        
        if channel_names is None:
            chanlocs = eeg_struct['chanlocs'][0]
            channel_names = [ch['labels'][0] for ch in chanlocs]
        
        eeg_list = []
        stimuli_list = []
        
        for trial_idx in range(len(csv_files)):
            trial_eeg = np.squeeze(eeg_trials[trial_idx])
            if trial_eeg.size == 0 or max(trial_eeg.shape) < (eeg_fs * 2) or len(trial_eeg.shape) < 2:
                continue
                
            if trial_eeg.shape[0] < trial_eeg.shape[1]: trial_eeg = trial_eeg.T
                
            current_csv_path = csv_files[trial_idx]
            csv_filename = os.path.basename(current_csv_path)
            env_filename = csv_filename.replace('ic_', '').replace('.csv', '.npy')
            env_path = os.path.join(env_dir, env_filename)
            
            if not os.path.exists(env_path): continue
            
            clean_stim, clean_eeg = prepare_song_trial(
                eeg_matrix=trial_eeg, env_path=env_path, ic_csv_path=current_csv_path, 
                eeg_fs=eeg_fs, env_dt=env_dt, ic_dt=ic_dt 
            )
            stimuli_list.append(clean_stim)
            eeg_list.append(clean_eeg)
            
        if not stimuli_list: continue
            
        print(f"  -> Fitting direct models for {subj_id}...")
        z_full, z_base, delta_z, opt_lam = compare_direct_models(stimuli_list, eeg_list, eeg_fs, use_banded=use_banded)
        
        # --- Collect null permutations for Group Stats ---
        _, z_null_perms = evaluate_with_mismatched_epochs(
            stimuli_list, eeg_list, eeg_fs, opt_lam, n_perms=n_perms, use_banded=use_banded
        )
        
        delta_z = np.nan_to_num(delta_z, nan=0.0)
        z_base = np.nan_to_num(z_base, nan=0.0)
        
        all_delta_z.append(delta_z)
        all_z_base.append(z_base)
        all_eeg_null_perms.append(z_null_perms)

    if not all_delta_z: raise RuntimeError("No subjects were successfully processed.")

    print("\n==========================================")
    print(" Computing Group-Level Permutation Statistics...")
    print("==========================================")

    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]

    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]
    
    # ---------------------------------------------------------
    # 1. Baseline (Envelope) Stats
    # ---------------------------------------------------------
    group_z_base = np.stack(all_z_base, axis=0)[:, eeg_indices] 
    mean_z_base = np.mean(group_z_base, axis=0)
    
    p_values_base = []
    for ch in range(group_z_base.shape[1]):
        try: _, p = jackknife_ttest(group_z_base[:, ch], alternative='greater')
        except Exception: p = wilcoxon(group_z_base[:, ch], alternative='greater').pvalue
        p_values_base.append(p)
        
    reject_null_base, p_values_fdr_base = mne.stats.fdr_correction(p_values_base, alpha=0.05)
    significant_channels_base = np.where(reject_null_base)[0]
    
    print(f"Overall Mean Scalp Envelope Prediction (z_base): {np.mean(mean_z_base):.4f}")
    print(f"Number of channels with significant envelope prediction: {len(significant_channels_base)} / {len(valid_eeg_names)}")
    
    if len(significant_channels_base) > 0:
        print("\nSignificant Envelope Channels (FDR Corrected):")
        for ch_idx in significant_channels_base:
            print(f" - {valid_eeg_names[ch_idx]}: z_base = {mean_z_base[ch_idx]:.4f} (p = {p_values_fdr_base[ch_idx]:.4f})")

    # ---------------------------------------------------------
    # 2. Permutation Stats for IC Enhancement
    # ---------------------------------------------------------
    group_delta_z = np.stack(all_delta_z, axis=0)[:, eeg_indices]
    mean_delta_z = np.mean(group_delta_z, axis=0)
    
    group_eeg_null_perms = np.stack(all_eeg_null_perms, axis=0)[:, :, eeg_indices]
    base_expanded = group_z_base[:, np.newaxis, :]
    null_delta_z_matrix = group_eeg_null_perms - base_expanded
    group_null_delta_z_dist = np.mean(null_delta_z_matrix, axis=0) 
    
    # Calculate true p-value: % of random shuffles that beat the true group mean
    p_values_perm = np.sum(group_null_delta_z_dist >= mean_delta_z, axis=0) / n_perms
    
    # --- NEGATIVE ANOMALY possible workaround ---
    #p_values_perm[mean_delta_z <= 0] = 1.0 
    
    reject_null_perm, p_values_fdr_perm = mne.stats.fdr_correction(p_values_perm, alpha=0.05)
    significant_channels_perm = np.where(reject_null_perm)[0]

    print(f"\nNumber of EEG channels with significant IC enhancement (Permutation FDR): {len(significant_channels_perm)} / {len(valid_eeg_names)}")
    
    if len(significant_channels_perm) > 0:
        print("\nSignificant IC Channels (Permutation FDR Corrected):")
        for ch_idx in significant_channels_perm:
            print(f" - {valid_eeg_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr_perm[ch_idx]:.4f})")

    return {
        "mean_delta_z": mean_delta_z,
        "p_values_raw": p_values_perm,
        "significant_mask": reject_null_perm,
        "channel_names": valid_eeg_names
    }

# ====================================================================
# ARG PARSER
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG TRF Analysis on Music Datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['openneuro', 'song'])
    parser.add_argument('--bids_dir', type=str, default="ds002725")
    parser.add_argument('--env_dir', type=str, default="enveloppe_of_stimuli")
    parser.add_argument('--ic_dir', type=str, default="ic_curves_global_stimuli")
    parser.add_argument('--mat_dir', type=str, default="song_data_mat")
    parser.add_argument('--csv_dir', type=str, default="../Giorgia_data/features/ic_neurips/aligned")
    parser.add_argument('--npy_dir', type=str, default="../Giorgia_data/features/64Hz/eel_envelope")
    parser.add_argument('--env_dt', type=float, default=0.015625)
    parser.add_argument('--ic_dt', type=float, default=0.7)
    parser.add_argument('--banded', action='store_true')
    parser.add_argument('--n_perms', type=int, default=100, help="Number of permutations for testing (default: 100)")

    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    if args.dataset == 'openneuro':
        print("\n=== Launching OpenNeuro Pipeline ===")
        results = run_openneuro_dataset_analysis(
            bids_root_dir=args.bids_dir, env_dir=args.env_dir, ic_dir=args.ic_dir,
            use_banded=args.banded, n_perms=args.n_perms
        )
        
        fig = plot_group_level_topomap(
            mean_scores=results["mean_delta_z"], ch_names=results["channel_names"],
            significant_mask=results["significant_mask"], score_type="Mean Δz", cmap="Reds", vmin=0.0
        )
        fig.savefig("results/group_level_enhancement_openneuro.png", dpi=300)
        print("Saved openneuro average topomap to results/group_level_enhancement_song.png")

        raw_p_vals = results["p_values_raw"]
        ch_names = results["channel_names"]
        mean_dz = results["mean_delta_z"]

        print("\n--- Uncorrected Permutation Enhancements (p < 0.05) ---")
        found_any = False
        for i, p in enumerate(raw_p_vals):
            if p < 0.05:
                print(f" - {ch_names[i]}: Δz = {mean_dz[i]:.4f} (perm p = {p:.4f})")
                found_any = True
        if not found_any: print(" No channels reached uncorrected significance.")

    elif args.dataset == 'song':
        print("\n=== Launching SONG Dataset Pipeline ===")
        results = run_song_dataset_analysis(
            mat_dir=args.mat_dir, env_dir=args.npy_dir, csv_dir=args.csv_dir,
            eeg_fs=64, env_dt=args.env_dt, ic_dt=args.ic_dt, use_banded=args.banded, n_perms=args.n_perms
        )
        
        fig = plot_group_level_topomap(
            mean_scores=results["mean_delta_z"], ch_names=results["channel_names"],
            significant_mask=results["significant_mask"], score_type="Mean Δz", cmap="Reds", vmin=0.0
        )
        fig.savefig("results/group_level_enhancement_song.png", dpi=300)
        print("Saved SONG average topomap to results/group_level_enhancement_song.png")

        raw_p_vals = results["p_values_raw"]
        ch_names = results["channel_names"]
        mean_dz = results["mean_delta_z"]

        print("\n--- Uncorrected Permutation Enhancements (p < 0.05) ---")
        found_any = False
        for i, p in enumerate(raw_p_vals):
            if p < 0.05:
                print(f" - {ch_names[i]}: Δz = {mean_dz[i]:.4f} (perm p = {p:.4f})")
                found_any = True
        if not found_any: print(" No channels reached uncorrected significance.")
