import os
import glob
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from pathlib import Path

def build_direct_dataset():
    print("Starting multi-noise dataset construction from raw files...")

    # =========================================================
    # 1. SET YOUR RAW DATA PATHS
    # =========================================================
    env_dir = Path("../Giorgia_data/features/64Hz/eel_envelope").resolve()
    
    # Base directory containing the three noise level subfolders
    #base_ic_dir = Path("/Users/augustin.bouquillard/Downloads/pa_audioic/song_pa-audioic_ic_0.0-0.3-0.5").resolve()
    base_ic_dir = Path("/Users/augustin.bouquillard/Downloads/pa_audioic/song_pa-audioic_entropy_mc32_nruns1_0.0-0.3-0.5").resolve()
    
    # Define paths for all three noise levels
    ic_dir_0 = base_ic_dir / "noise_0"
    ic_dir_3 = base_ic_dir / "noise_0.3"
    ic_dir_5 = base_ic_dir / "noise_0.5"

    env_dt = 0.015625
    ic_dt = 0.0928798
    # =========================================================

    # We use noise_0 to determine the alphabetical sorting order, maintaining the exact EEG alignment
    ic_files_0 = sorted(
        glob.glob(str(ic_dir_0 / "ic_*_song.csv")) +
        glob.glob(str(ic_dir_0 / "*_song.npy"))
    )

    if not ic_files_0:
        print(f"Error: No IC files found in {ic_dir_0}")
        return

    data_dict = {
        "id": [],
        "envelope": [],
        "ic_noise_0_0": [],  
        "ic_noise_0_3": [],
        "ic_noise_0_5": [],
        "ic_time": []        
    }

    print(f"Found {len(ic_files_0)} IC files. Processing all three noise levels...")

    for trial_idx, ic_path_0 in enumerate(ic_files_0):
        filename = os.path.basename(ic_path_0)

        # Reconstruct the expected paths for the other two noise levels
        ic_path_3 = ic_dir_3 / filename
        ic_path_5 = ic_dir_5 / filename

        # Match the acoustic envelope
        base_name = filename.replace('ic_', '').rsplit('.', 1)[0]
        env_filename = base_name + '.npy'
        env_path = env_dir / env_filename

        if not env_path.exists():
            print(f"  [Skipping] {filename}: Matching envelope {env_filename} not found.")
            continue

        # Load envelope as 64-bit float
        env_data = np.squeeze(np.load(env_path)).astype(np.float64)

        # Helper function to load IC data regardless of file type
        def load_ic_data(path):
            if str(path).endswith('.npy'):
                return np.squeeze(np.load(path)).astype(np.float64)
            else:
                ic_df = pd.read_csv(path)
                return ic_df.iloc[:, 1].fillna(0.0).values.astype(np.float64)

        # Load all three noise conditions
        ic_data_0 = load_ic_data(ic_path_0)
        ic_data_3 = load_ic_data(ic_path_3)
        ic_data_5 = load_ic_data(ic_path_5)

        # Establish the time vector using noise_0 as the reference
        if filename.endswith('.npy'):
            ic_time = np.arange(len(ic_data_0)) * ic_dt
        else:
            ic_df_0 = pd.read_csv(ic_path_0)
            if 'time' in ic_df_0.columns:
                ic_time = ic_df_0['time'].values.astype(np.float64)
            else:
                ic_time = np.arange(len(ic_data_0)) * ic_dt

        # Append to dictionary using the raw integer index for PyArrow
        data_dict["id"].append(trial_idx)
        data_dict["envelope"].append(env_data)
        data_dict["ic_noise_0_0"].append(ic_data_0)
        data_dict["ic_noise_0_3"].append(ic_data_3)
        data_dict["ic_noise_0_5"].append(ic_data_5)
        data_dict["ic_time"].append(ic_time)

    # Use int64 for the ID schema and include all three noise columns
    features = Features({
        'id': Value('int64'),
        'envelope': Sequence(Value('float64')),
        'ic_noise_0_0': Sequence(Value('float64')),
        'ic_noise_0_3': Sequence(Value('float64')),
        'ic_noise_0_5': Sequence(Value('float64')),
        'ic_time': Sequence(Value('float64'))
    })

    ds = Dataset.from_dict(data_dict, features=features)

    try:
        root_dir = Path(__file__).resolve().parent.parent.parent
    except NameError:
        root_dir = Path.cwd()

    #out_file = root_dir / "hf_datasets_mex" / "song_pa-audioic_ic-0.0-0.3-0.5" / "train.parquet"
    out_file = root_dir / "hf_datasets_mex" / "song_pa-audioic_entropy_mc32_nruns1_0.0-0.3-0.5" / "train.parquet"
    print(f"Saving multi-noise dataset to: {out_file}")
    
    ds.to_parquet(str(out_file))
    print("\nDataset successfully built!")

if __name__ == "__main__":
    build_direct_dataset()
