import numpy as np
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval

#I could cross validate correlations between EEG and stimulus waveform to find the best hardware delay, but there may be no need with TRF

def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """Applies Fisher z-transformation to Pearson's r values to normalize variance."""
    r_clipped = np.clip(r, -0.999, 0.999)
    return np.arctanh(r_clipped)

def compare_direct_models(stimuli_list, eeg_list, fs):#, lambda_val=1.0):
    """
    Method 1: simple comparison between baseline (energy only) and full (energy + IC).

    Parameters:
    - stimuli_list: list of 2D arrays (n_samples, 2). Col 0 = E, Col 1 = IC.
    - eeg_list: list of 2D arrays (n_samples, n_channels).

    Returns: z_full, z_base, delta_z (all shape: n_channels,)
    """
    tmin, tmax = -0.1, 0.7 #time window [-100,700]ms
    k_folds = -1  # Leave-One-Out CV
    regularization = np.logspace(-1, 4, 6)
    trf = TRF(direction=1)

    # baseline stimuli (Energy Only - column 0)
    baseline_stimuli = [stim[:, 0:1] for stim in stimuli_list]#0:1 slicing keeps it a 2D array, which mTRFpy expects

    # baseline model (E only)
    print("Fitting Baseline Model (Energy only)...") 
    r_base, best_lambdas_base = nested_crossval(trf, baseline_stimuli, eeg_list, fs, tmin, tmax, regularization, k=k_folds, average=True, verbose=False)
    opt_lambda = np.median(best_lambdas_base)
    r_base = crossval(trf, baseline_stimuli, eeg_list, fs, tmin, tmax, opt_lambda, k=k_folds, average=False, verbose=False)
    if r_base.ndim==2:
        print("averaging r across trials but not across channels (average is False in cross_val")
        r_base=np.mean(r_base, axis=0)
    z_base = fisher_z_transform(r_base)
    print(f'Best energy only model regularization parameter: {opt_lambda}')

    # full model (E + IC) #TO DO : have a different regularization lambda for each feature would make our claims stronger 
    print("Fitting Full Model (Energy + IC)...")
    r_full, best_lambdas_full = nested_crossval(trf, stimuli_list, eeg_list, fs, tmin, tmax, regularization, k=k_folds, average=True, verbose=False)
    opt_lambda = np.median(best_lambdas_full)
    r_full = crossval(trf, stimuli_list, eeg_list, fs, tmin, tmax, opt_lambda, k=k_folds, average=False, verbose=False)
    if r_full.ndim==2:
        print("averaging r across trials")
        r_full=np.mean(r_full, axis=0)
    z_full = fisher_z_transform(r_full)
    print(f'Best full model regularization parameter: {opt_lambda}')

    # predictive enhancement
    delta_z = z_full - z_base

    return z_full, z_base, delta_z, opt_lambda


def evaluate_with_circular_shift(stimuli_list, eeg_list, fs, lambda_val=1.0, n_perms=100, min_shift_sec=10.0, nested=True):
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
    
    # apply standard 10-20 scalp montage TO DO : get the montage from the data itself !
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
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
        vlim=(vmin, vmax), # Locks the color scale if provided
        show=False,
        extrapolate='local' # Prevents bleeding outside the head circle
    )
    
    ax.set_title(f"Spatial Distribution of Model Accuracy\nParticipant: {subj_id}")
    
    # Add a colorbar to interpret the values
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
    times_to_plot: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
):
    # Extract weights and transpose for MNE
    weights = trf_model.weights[feature_idx, :, :]
    weights_mne = weights.T 
    
    # Create MNE Info and montage
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, match_case=False, on_missing='warn')
    except ValueError:
        pass

    evoked = mne.EvokedArray(weights_mne, info, tmin=tmin)
    
    fig = evoked.plot_topomap(
        times=times_to_plot,
        ch_type='eeg',
        time_unit='s',
        colorbar=True,
        show=False 
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
        montage = mne.channels.make_standard_montage('standard_1020')
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
        extrapolate='local'
    )
    
    ax.set_title("Group-Level Predictive Enhancement\n(White dots = FDR corrected p < 0.05)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(score_type)
    
    fig.tight_layout()
    return fig



#######################################################################################################################
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
    eeg_length = raw.n_times
    
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
            ic_time, ic_data, kind='linear', bounds_error=False, fill_value=0.0
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
    Loops through an OpenNeuro BIDS dataset, aligns data, runs TRF, 
    and computes group-level summary statistics.
    """
    subject_folders = sorted(glob.glob(os.path.join(bids_root_dir, "sub-*")))
    if not subject_folders:
        raise ValueError(f"No BIDS sub-* folders found in {bids_root_dir}")
        
    all_delta_z = []
    channel_names = None
    eeg_fs = None 
    
    # Standard 10-20 channels used for mapping (excluding joysticks and ECG)
    eeg_only_names = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
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
            # I skip the generated music tasks for now
            if "genMusic" in edf_file:
                continue

            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            if eeg_fs is None:
                eeg_fs = raw.info['sfreq']
            
            try:
                # aligning stimuli with the 'music' trigger channel
                aligned_stim = choose_stimulus_and_align_to_eeg(
                    raw=raw, 
                    env_dir=env_dir, 
                    ic_dir=ic_dir, 
                    env_dt=env_dt, 
                    ic_dt=ic_dt
                )
                
                # dropping technical channels (keeping joysticks/ECG for the TRF)
                technical_channels = ['music', 'trialtype', 'nback_stimuli', 'nback_keypress', 'POz']
                channels_to_drop = [ch for ch in technical_channels if ch in raw.ch_names]
                if channels_to_drop:
                    raw.drop_channels(channels_to_drop)

                # remaining channels (EEG + Misc/Joysticks)
                picks = mne.pick_types(raw.info, eeg=True, misc=True, stim=True)
                
                # list of channels being fed to the TRF
                if channel_names is None:
                    channel_names = [raw.ch_names[i] for i in picks]
                    
                # Extract EEG data matrix (Transpose for mTRFpy -> Time x Channels)
                raw_eeg_trial = raw.get_data(picks=picks).T 
                
                valid_indices = np.where(aligned_stim[:, 0] > 0)[0]
                
                if len(valid_indices) == 0:
                    print("  [!] No valid stimuli data found. Skipping EDF.")
                    continue
                    
                # Find gaps in the valid indices to separate songs
                split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
                valid_blocks = np.split(valid_indices, split_points)
                
                for block in valid_blocks:
                    # Ensure the block is at least 10 seconds long
                    if len(block) > (10 * eeg_fs):
                        chunk_size = int(30 * eeg_fs) # 30-second epochs
                        
                        for i in range(0, len(block), chunk_size):
                            chunk_idx = block[i : i + chunk_size]
                            
                            # Ensure the final chunk isn't tiny (must be at least 15s)
                            if len(chunk_idx) > (15 * eeg_fs):
                                # Z-score to standardize scale, ignoring flat-line warnings
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    chunk_stim = np.nan_to_num(zscore(aligned_stim[chunk_idx, :], axis=0))
                                    chunk_eeg = np.nan_to_num(zscore(raw_eeg_trial[chunk_idx, :], axis=0))
                                
                                stimuli_list.append(chunk_stim)
                                eeg_list.append(chunk_eeg)
                                
            except ValueError as e:
                print(f"  [!] {e} Skipping EDF.")
                continue
            
        # ====================================================================
        # Subject-Level TRF Pipeline
        if not stimuli_list:
            print(f"Skipping {subj_id} TRF: No valid epoched trials available.")
            continue
            
        try:
            print(f"  -> Fitting Direct Models (Cross-Validation on {len(stimuli_list)} epochs)...")
            z_full, z_base, delta_z, opt_lambda = compare_direct_models(stimuli_list, eeg_list, eeg_fs)
            delta_z = np.nan_to_num(delta_z, nan=0.0)
            all_delta_z.append(delta_z)
            print(f"[{subj_id}] Mean channel enhancement: {np.mean(delta_z):.4f}")

            #plotting only the valid eeg channels
            eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
            valid_eeg_names = [channel_names[i] for i in eeg_indices]
            scalp_scores = delta_z[eeg_indices]

            os.makedirs("results", exist_ok=True)

            # Static Topomap (scalp only)
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

            # Fit Final Full Model on all data to extract Spatiotemporal weights
            print(f"  -> Fitting final Full Model to extract weights...")
            all_stimuli_concat = np.vstack(stimuli_list)
            all_eeg_concat = np.vstack(eeg_list)

            final_trf = TRF(direction=1)
            opt_lam_full = opt_lambda

            final_trf.train(
                all_stimuli_concat, 
                all_eeg_concat, 
                eeg_fs, 
                tmin=-0.1, 
                tmax=0.7, 
                regularization=opt_lam_full
            )

            # Slice the TRF weights matrix so it only contains the scalp channels
            # mTRFpy weights shape: (n_features, n_lags, n_channels)
            sliced_weights = final_trf.weights[:, :, eeg_indices]
            final_trf.weights = sliced_weights 

            fig = plot_participant_trf_topomaps(
                trf_model=final_trf,
                ch_names=valid_eeg_names, 
                fs=eeg_fs,
                tmin=-0.1,
                subj_id=subj_id,
                feature_idx=1, # Column 1 is IC
                feature_name="Information Content (IC)",
                times_to_plot=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5] 
            )
            fig.savefig(f"results/topomap_{subj_id}_IC_weights.png", dpi=300)
            plt.close(fig)

        except Exception as e:
            print(f"Failed to process {subj_id} TRF: {e}")
            continue

    # ====================================================================
    # Group-Level Statistics
    if not all_delta_z:
        raise RuntimeError("No subjects were successfully processed.")

    print("\n==========================================")
    print(" Computing Group-Level Statistics...")
    print("==========================================")
    
    group_delta_z = np.stack(all_delta_z, axis=0) 
    mean_delta_z = np.mean(group_delta_z, axis=0)
    std_delta_z = np.std(group_delta_z, axis=0)
    
    t_stats, p_values = ttest_1samp(group_delta_z, popmean=0.0, axis=0, alternative='greater')
    reject_null, p_values_fdr = mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')
    significant_channels = np.where(reject_null)[0]
    
    print(f"Total Subjects Processed: {group_delta_z.shape[0]}")
    print(f"Overall Mean Enhancement: {np.mean(mean_delta_z):.4f}")
    
    # We can print significant findings for ALL channels (including joysticks!)
    print(f"Number of channels with significant enhancement: {len(significant_channels)} / {len(channel_names)}")
    
    if len(significant_channels) > 0:
        print("\nSignificant Channels (FDR Corrected):")
        for ch_idx in significant_channels:
            print(f" - {channel_names[ch_idx]}: Δz = {mean_delta_z[ch_idx]:.4f} (p = {p_values_fdr[ch_idx]:.4f})")

    return {
        "mean_delta_z": mean_delta_z,
        "std_delta_z": std_delta_z,
        "p_values_raw": p_values,
        "p_values_fdr": p_values_fdr,
        "significant_mask": reject_null,
        "channel_names": channel_names
    }
                        
 




if  __name__ == "__main__":
    
    bids_path = "ds002725"

    results = run_group_level_analysis(bids_path, "enveloppe_of_stimuli", "ic_curves_global_stimuli")
    
    # standard scalp channels to keep
    eeg_only_names = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
    ]
    
    channel_names = results["channel_names"]
    
    # indices of just the scalp channels
    eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
    valid_eeg_names = [channel_names[i] for i in eeg_indices]
    

    scalp_mean_z = results["mean_delta_z"][eeg_indices]
    
    if results["significant_mask"] is not None:
        scalp_mask = results["significant_mask"][eeg_indices]
    else:
        scalp_mask = None

    fig = plot_group_level_topomap(
        mean_scores=scalp_mean_z,
        ch_names=valid_eeg_names,
        significant_mask=scalp_mask,
        score_type="Mean Δz",
        cmap="Reds",
        vmin=0.0
    )
    
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/group_level_enhancement.png", dpi=300)
    print("Saved group-level topomap to results/group_level_enhancement.png")
    
