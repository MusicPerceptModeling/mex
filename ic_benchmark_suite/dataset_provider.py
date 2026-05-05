from dataclasses import dataclass
from pathlib import Path
import datasets
from huggingface_hub import HfApi
from platformdirs import user_cache_dir

@dataclass
class DatasetProvider:
    cache_dir : str =  None
    hf_root: str = 'muthissar/music_surprisal_benchmarks'
    hf_token : str = None
    def get_ds(self, ds_name: str = 'session_ds') -> datasets.Dataset:
        try:
            return datasets.load_dataset(self.hf_root, ds_name, cache_dir=self.cache_dir, token=self.hf_token, split='train')
        except Exception as e:
            # datasets.Dataset.from_parquet(Path(self.cache_dir, ds_name, 'train.parquet'))
            # return datasets.load_from_disk(Path(self.cache_dir, ds_name))
            return datasets.load_from_disk(ds_name)
        except Exception as e:
            raise ValueError(f"Could not load dataset {ds_name} from Hugging Face Hub or local cache. Error: {e}")
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = user_cache_dir('mex')
        # Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    def persist_ds(self, ds : datasets.Dataset, ds_name: str = 'session_ds', hub_upload: bool = True):
        if hub_upload:
            file = Path(ds_name, 'train.parquet')
            local_file = self.cache_dir / file
            ds.to_parquet(local_file)
            # Upload each Parquet file to the Hub
            api = HfApi(token=self.hf_token)
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=str(file),
                repo_id=self.hf_root,
                repo_type="dataset"
            )
        else:
            # ds.to_parquet(Path(self.cache_dir, ds_name, 'train.parquet'))
            ds.save_to_disk(ds_name)