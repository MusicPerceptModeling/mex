from pathlib import Path
import datasets
from datasets.features import Value, Sequence, Features

# 1. Define the dataset name and load it from your local cache
ds_name = 'song_audioic_entropy_mc32' 
ds = datasets.load_from_disk(f'cache/metrics/{ds_name}')
print("Original dataset schema:", ds)

# 2. Resolve the ID issue: Convert integer IDs to strings
# This ensures it matches the 'stimulus_id' (e.g., "trial_01") in the EEG metadata
if ds.features['id'].dtype != 'string':
    print("Converting 'id' column to string...")
    # Add prefix if needed, e.g., 'trial_' + str(x['id']) depending on your EEG mat files
    ds = ds.map(lambda x: {'id': f"trial_{int(x['id']):02d}"}) 

#Define the schema required by the Neural Encoding Task
#Ensure these keys exactly match the column names in your loaded `ds`
features = Features({
    'id': Value('string'),
    'envelope': Sequence(Value('float32')),
    'entropy': Sequence(Value('float32')), # 'ic' if the column of interest in the dataset to convert is named 'ic'
})

# 4. Select only the necessary columns and cast to float32 to save space
casted = ds.select_columns(list(features.keys())).cast(features)
print("\nCasted dataset ready for export:", casted)

# 5. Save to the MEX huggingface folder structure as a Parquet file
new_folder = Path('hf_datasets_mex', ds_name)
new_folder.mkdir(parents=True, exist_ok=True)
out = new_folder / 'train.parquet'
casted.to_parquet(out)

print(f"\nSuccessfully saved optimized MIDM dataset to {out}")
print(f"Don't forget to update hf_datasets_mex/README.md to expose {ds_name}!")
