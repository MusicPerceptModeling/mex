from pathlib import Path
import re
from typing import Callable

from jsonargparse import CLI
from dataclasses import dataclass

from ic_benchmark_suite import idyom_ds
import datasets

from ic_benchmark_suite.baseline import Baseline
from ic_benchmark_suite import idyom_corr
from ic_benchmark_suite.benchmark_repetition_test import _salami_parsed_features
from ic_benchmark_suite.dataset_provider import DatasetProvider


def extract_numeric_stem_id(path: str) -> int:
    return int(re.search(r"\d+", Path(path).stem).group())

@dataclass
class BenchmarkSuite:
    dataset_provider: DatasetProvider
    def create_idyom_ds(
        self,
        ds_name: str,
        session_dat_dir: str,
        midi_dir : str,  
        n_midis : int = 200,
        num_proc : int = 60,
        hub_upload: bool = False,
        id_extractor : Callable[[str], int] =  extract_numeric_stem_id
    ):
        session_idyom_ds = idyom_ds(
            Path(session_dat_dir),
            midi_dir=Path(midi_dir),
            id_path_extractor = id_extractor,
            cache_dir=self.dataset_provider.cache_dir,
            n_midis=n_midis,
            num_proc=num_proc,
        )
        self.dataset_provider.persist_ds(session_idyom_ds, ds_name=ds_name, hub_upload=hub_upload)
    def baseline(self, model: Baseline, audio_ds : str | datasets.Dataset , cache_name : str, val: str = 'ic', hub_upload: bool = False):
        if isinstance(audio_ds, str):
            audio_ds = self.dataset_provider.get_ds(audio_ds)
        if val == 'ic':
            ic_ds = model.ic(audio_ds)
        elif val == 'entropy':
            ic_ds = model.entropy(audio_ds)
        self.dataset_provider.persist_ds(ic_ds, ds_name=cache_name, hub_upload=hub_upload)

    def benchmark(self, task: str, cache_model: str, ds_name: str, task_kwargs: dict = {}):
        model_outputs = self.dataset_provider.get_ds(cache_model)
        task_ds = self.dataset_provider.get_ds(ds_name)
        if task == 'idyom-consistency':
            if 'audios' in task_ds.column_names:
                task_ds = task_ds.remove_columns(['audios'])
            ic_cols = [col for col in model_outputs.column_names if col.startswith('ic')]
            entr_cols = [col for col in model_outputs.column_names if col.startswith('entr')]
            assert not (ic_cols and entr_cols), "IC dataset cannot contain columns starting with 'ic' and 'entr'."
            if entr_cols:
                task_ds = task_ds.remove_columns(['idyom_ic'])
                labels = entr_cols
            elif ic_cols:
                task_ds = task_ds.remove_columns(['idyom_entropy'])
                labels = ic_cols

            idyom_df = task_ds.to_pandas().set_index('id')
            results =  idyom_corr(
                val_dataset=model_outputs,
                idyom_df=idyom_df,
                labels=labels,
                **task_kwargs
            )
            print(results)
        elif task == 'repetition':
            task_features = _salami_parsed_features()
            # task_ds = task_ds.cast(task_features)
            assert task_ds.features == task_features

if __name__ == '__main__':
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
    CLI(BenchmarkSuite, as_positional=True, default_config_files=[str(config_path)])