from pathlib import Path
import re
from typing import Callable

from jsonargparse import CLI
from dataclasses import dataclass

from ic_benchmark_suite import idyom_ds
import datasets

from ic_benchmark_suite.baseline import Baseline
from ic_benchmark_suite import idyom_corr
from ic_benchmark_suite.benchmark_repetition_test import repetition_decrease
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
            try:
                # TODO: make this self explainable. Deal with how arguments should be passed. Alternativey, we should make seperate methods for each task.
                val = task_kwargs.pop('value')
                task_ds = task_ds.select_columns(['id', 'onset', f'idyom_{val}']).rename_column(f'idyom_{val}', 'idyom_val')
            except Exception as e:
                raise ValueError("For idyom-consistency task, task_kwargs must contain 'value' key with value 'ic' or 'entropy'.") from e

            # TODO: handle time_offset_frame for ic/entropy.
            results =  idyom_corr(
                val_dataset=model_outputs,
                idyom_ds=task_ds,
                **task_kwargs
            )
            print(results)
        elif task == 'repetition':
            results = repetition_decrease(
                model_outputs=model_outputs,
                metadata_ds=task_ds,
                **task_kwargs
            )
            print({label: {annotation: res[annotation][0] for annotation in res} for label, res in results.items()})
if __name__ == '__main__':
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
    CLI(BenchmarkSuite, as_positional=True, default_config_files=[str(config_path)])