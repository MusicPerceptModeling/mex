import os
import glob
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from pathlib import Path

def build_direct_dataset():
    print("Starting direct dataset construction from raw files...")

    env_dir = Path("../Giorgia_data/features/64Hz/eel_envelope").resolve()
    ic_dir = Path("/Users/augustin.bouquillard/Downloads/pa_audioic/song_pa-audioic_ic_0.0-0.3-0.5/noise_0").resolve()

    env_dt = 0.015625
    ic_dt = 0.0928798

    # Alphabetical sort to match the EEG arrays perfectly
    ic_files = sorted(
        glob.glob(str(ic_dir / "ic_*_song.csv")) +
        glob.glob(str(ic_dir / "*_song.npy"))
    )

    if not ic_files:
        print(f"Error: No IC files found in {ic_dir}")
        return

    data_dict = {
        "id": [],
        "envelope": [],
        "ic_noise_0_0": [],  
        "ic_time": []        
    }

    print(f"Found {len(ic_files)} IC files. Processing...")

    for trial_idx, ic_path in enumerate(ic_files):
        filename = os.path.basename(ic_path)

        base_name = filename.replace('ic_', '').rsplit('.', 1)[0]
        env_filename = base_name + '.npy'
        env_path = env_dir / env_filename

        if not env_path.exists():
            print(f"  [Skipping] {filename}: Matching envelope {env_filename} not found.")
            continue

        env_data = np.squeeze(np.load(env_path)).astype(np.float64)

        if filename.endswith('.npy'):
            ic_data = np.squeeze(np.load(ic_path)).astype(np.float64)
            ic_time = np.arange(len(ic_data)) * ic_dt
        else:
            ic_df = pd.read_csv(ic_path)
            ic_data = ic_df.iloc[:, 1].fillna(0.0).values.astype(np.float64)
            if 'time' in ic_df.columns:
                ic_time = ic_df['time'].values.astype(np.float64)
            else:
                ic_time = np.arange(len(ic_data)) * ic_dt

        # Use the raw integer index to pass PyArrow's int64 requirement
        data_dict["id"].append(trial_idx)
        data_dict["envelope"].append(env_data)
        data_dict["ic_noise_0_0"].append(ic_data)
        data_dict["ic_time"].append(ic_time)

    # Use int64 for the ID schema
    features = Features({
        'id': Value('int64'),
        'envelope': Sequence(Value('float64')),
        'ic_noise_0_0': Sequence(Value('float64')),
        'ic_time': Sequence(Value('float64'))
    })

    ds = Dataset.from_dict(data_dict, features=features)

    try:
        root_dir = Path(__file__).resolve().parent.parent.parent
    except NameError:
        root_dir = Path.cwd()

    out_dir = root_dir / "hf_datasets_mex" / "song_midm_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving pristine dataset to: {out_dir}")
    ds.save_to_disk(str(out_dir))
    print("\nDataset successfully built!")

if __name__ == "__main__":
    build_direct_dataset()
