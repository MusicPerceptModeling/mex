from dataclasses import dataclass
import os
from pathlib import Path
import datasets
from platformdirs import user_cache_dir

from mex.util.hf import update_readme_config_entry


@dataclass
class DatasetProvider:
    cache_dir : str =  None
    #hf_root: str = 'muthissar/music_surprisal_benchmarks'
    hf_root: str = "MusicPerceptModeling/mex"
    hf_token : str = None
    hf_revision: str = None
    hf_upload: bool = False
    # TODO: maybe only use use_cache, the other way around, we can manually load a dataset
    def get_ds(self, ds_name: str, use_cache=True) -> datasets.Dataset:
        try:
            return datasets.load_dataset(self.hf_root, ds_name, cache_dir=self.cache_dir, token=self.hf_token, split='train', revision=self.hf_revision)
        except Exception as e:
            # TODO: it might be dangeruous to use local files, especially cache in new versions of the code, since versions are not checked.
            ds_name = f"{self.cache_dir}/{ds_name}" if use_cache else ds_name
            return datasets.load_from_disk(ds_name)
        except Exception as e:
            raise ValueError(f"Could not load dataset {ds_name} from Hugging Face Hub or local cache. Error: {e}")
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = user_cache_dir('mex')
        if self.hf_token is None and 'HF_TOKEN' in os.environ:
            self.hf_token = os.environ.get('HF_TOKEN')
        # Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    def persist_ds(self, ds : datasets.Dataset, ds_name: str, use_cache=True):
        if self.hf_upload:
            # NOTE: managed by gitsubmodule
            path_within_repo = ds_name
            local_file = Path('hf_datasets_mex', path_within_repo, 'train.parquet')
            ds.to_parquet(local_file)
            update_readme_config_entry(config_name=ds_name, data_dir=path_within_repo, readme_path=Path('hf_datasets_mex', 'README.md'))
            
        else:
            # ds.to_parquet(Path(self.cache_dir, ds_name, 'train.parquet'))
            ds_name = f"{self.cache_dir}/{ds_name}" if use_cache else ds_name
            ds.save_to_disk(ds_name)
