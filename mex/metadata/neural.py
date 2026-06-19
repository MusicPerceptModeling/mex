from dataclasses import dataclass
import os
import glob
import numpy as np
import scipy.io as sio
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence
from ..benchmark import MetaData

class NeuralEncodingMetaData(MetaData):
    @classmethod
    def parse_metadata_scheme(cls, dataset: Dataset):
        scheme = cls._metadata_scheme()
        missing = set(scheme.keys()) - set(dataset.column_names)
        if missing:
            raise ValueError(f"Dataset is missing required columns for Neural Encoding: {missing}")
        
        # Optionally, cast the dataset to strictly ensure typing
        # dataset = dataset.cast(scheme)

    @classmethod
    def _metadata_scheme(cls) -> Features:
        return Features({
            "id": Value("string"),           # Unique row ID: e.g., "sub-01_song-05"
            "stimulus_id": Value("string"),  # ID of the music track: e.g., "song-05" (Matches midm 'id')
            "subject": Value("string"),      # Subject identifier: e.g., "sub-01"
            "eeg_data": Sequence(Sequence(Value("float32"))), # 2D Array: (time, channels)
            "ch_names": Sequence(Value("string")),            # Standardized channel names
            "fs": Value("float32")           # EEG Sampling frequency
        })


class SongMetaData(NeuralEncodingMetaData):
    def __init__(self, mat_dir: str = None, **kwargs):
        super().__init__()
        self.mat_dir = mat_dir

    @property
    def ds_name(self):
        return 'song_eeg_metadata'

    def _preprocess(self, cache_dir: str = None) -> Dataset:
        raise NotImplementedError(
            "This dataset is pre-computed. Load the parquet file directly via the HuggingFace Hub or local cache."
        )
    """
    def _preprocess(self, cache_dir: str = None) -> Dataset:
        #Reads local SONG .mat files, filters 30 specific scalp channels,
        #and packages them into a HuggingFace Dataset.
        assert self.mat_dir is not None, "mat_dir must be provided for preprocessing SongMetaData"
        
        mat_files = sorted(glob.glob(os.path.join(self.mat_dir, "pre_dataSub*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No matching .mat files found in '{self.mat_dir}'.")

        eeg_only_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 
            'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10'
        ]

        records = []

        for mat_file in mat_files:
            # Extract Subject ID (e.g., "Sub1")
            subj_id = os.path.basename(mat_file).split('.')[0].replace('pre_data', '')
            
            print(f"Parsing {subj_id}...")
            
            mat_data = sio.loadmat(mat_file)
            eeg_struct = mat_data['eeg'][0, 0]
            
            fs = float(eeg_struct['fs'].item())
            
            # Extract and align channel names
            chanlocs = eeg_struct['chanlocs'][0]
            channel_names = [ch['labels'][0] for ch in chanlocs]
            
            eeg_indices = [channel_names.index(ch) for ch in eeg_only_names if ch in channel_names]
            valid_eeg_names = [channel_names[i] for i in eeg_indices]
            
            eeg_trials = eeg_struct['data'][0]
            
            # Process each trial (assuming trial_idx strictly maps to a specific song sequence)
            for trial_idx, trial_eeg in enumerate(eeg_trials):
                # Skip invalid trials
                if trial_eeg.size == 0 or max(trial_eeg.shape) < (fs * 2):
                    continue
                    
                trial_eeg = np.squeeze(trial_eeg)
                if len(trial_eeg.shape) < 2:
                    continue
                    
                # Ensure shape is (time, channels)
                if trial_eeg.shape[0] < trial_eeg.shape[1]:
                    trial_eeg = trial_eeg.T
                    
                # Subset to the 30 specific EEG scalp channels
                trial_eeg = trial_eeg[:, eeg_indices]
                
                # Create the structured record
                record = {
                    "id": f"{subj_id}_trial_{trial_idx:02d}",
                    "stimulus_id": f"trial_{trial_idx:02d}", # This will be used to link to the MIDM dataset
                    "subject": subj_id,
                    "eeg_data": trial_eeg.astype(np.float32).tolist(), # Convert to native list for HF
                    "ch_names": valid_eeg_names,
                    "fs": fs
                }
                records.append(record)

        # Build and return the HuggingFace Dataset
        return Dataset.from_list(records, features=self._metadata_scheme())
        """
