from pathlib import Path
import re
from typing import Callable, List

from jsonargparse import CLI
from dataclasses import dataclass

from ic_benchmark_suite.metadata.symbolic import SessionIDyOMMetaData
import datasets

from ic_benchmark_suite.baseline import Baseline
from ic_benchmark_suite import idyom_corr
from ic_benchmark_suite.benchmark import MetaData, Task
from ic_benchmark_suite.benchmark_repetition_test import RepetitionDecreaseTask
from ic_benchmark_suite.benchmark_segment_boundary import segment_boundary_alignment
from ic_benchmark_suite.dataset_provider import DatasetProvider
from ic_benchmark_suite.metadata.segment import SalamiMetaData


def extract_numeric_stem_id(path: str) -> int:
    return int(re.search(r"\d+", Path(path).stem).group())

@dataclass
class BenchmarkSuite:
    dataset_provider: DatasetProvider
    # def create_idyom_ds(
    #     self,
    #     ds_name: str,
    #     session_dat_dir: str,
    #     midi_dir : str,  
    #     n_midis : int = 200,
    #     num_proc : int = 60,
    #     hub_upload: bool = False,
    #     id_extractor : Callable[[str], int] =  extract_numeric_stem_id
    # ):
    #     session_idyom_ds = SessionIDyOMMetaData.idyom_ds(
    #         idyom_dat_dir=Path(session_dat_dir),
    #         midi_dir=Path(midi_dir),
    #         id_path_extractor = id_extractor,
    #         cache_dir=self.dataset_provider.cache_dir,
    #         n_midis=n_midis,
    #         num_proc=num_proc,
    #     )
    #     self.dataset_provider.persist_ds(session_idyom_ds, ds_name=ds_name, hub_upload=hub_upload)
    # def create_salami_ds(self, ds_name : str, salami_root: str, n_proc: int | None, hub_upload: bool = False):
    #     salami_ds = SalamiMetaData.read_annotations_hf(
    #         folder=salami_root,
    #         cache_dir=self.dataset_provider.cache_dir,
    #         keep_ids=True,
    #         n_proc=n_proc)
    #     self.dataset_provider.persist_ds(salami_ds, ds_name=ds_name, hub_upload=hub_upload)
    def preprocess_metadata(self, metadata: MetaData):
        metadata.preprocess()
        
    def baseline(self, model: Baseline, audio_ds : str | datasets.Dataset , cache_name : str, val: str = 'ic', hub_upload: bool = False):
        if isinstance(audio_ds, str):
            audio_ds = self.dataset_provider.get_ds(audio_ds)
        if val == 'ic':
            ic_ds = model.ic(audio_ds)
        elif val == 'entropy':
            ic_ds = model.entropy(audio_ds)
        self.dataset_provider.persist_ds(ic_ds, ds_name=cache_name, hub_upload=hub_upload)

    def benchmark_2(self, task: Task, midm: str | List[str], metadata: MetaData | str, task_kwargs: dict = {}, outpath: str = None, print_result: bool = True):
        if isinstance(midm, str):
            midm = self.dataset_provider.get_ds(midm)
        else:
            midm_ = [self.dataset_provider.get_ds(midm[0])]
            for cm in midm[1:]:
                midm_.append(self.dataset_provider.get_ds(cm).remove_columns(['Time', 'id'])) # TODO: remove time
            midm = datasets.concatenate_datasets(midm_, axis=1)
        if isinstance(metadata, str):
            metadata = self.dataset_provider.get_ds(metadata)
        elif isinstance(metadata, MetaData):
            metadata = metadata.load()
        # TODO: possibly but the task_kwargs in the constructor instead
        result = task.run_task(midm=midm, metadata=metadata, **task_kwargs)
        if outpath is not None:
            result.save(path=outpath)
        if print_result:
            print(result)
    

    def benchmark(self, task: str, cache_model: str | List[str], ds_name: str, task_kwargs: dict = {}):
        if isinstance(cache_model, str):
            model_outputs = self.dataset_provider.get_ds(cache_model)
        else:
            model_outputs = [self.dataset_provider.get_ds(cache_model[0])]
            for cm in cache_model[1:]:
                model_outputs.append(self.dataset_provider.get_ds(cm).remove_columns(['Time', 'id'])) # TODO: remove
            model_outputs = datasets.concatenate_datasets(model_outputs, axis=1)
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
            model_outputs = model_outputs.remove_columns(['Time']) # TODO: remove
            # TODO: we could create it with the task parameters such that it can more easily be passed to 
            # TODO: we should do this regardless of the parameters to get rid of the  if clause
            repetition_result = RepetitionDecreaseTask().run_task(
                midm=model_outputs,
                metadata=task_ds,
                **task_kwargs
            )
            print(repetition_result.summary)
        elif task == "segment-boundary-alignment":
            results = segment_boundary_alignment(
                model_outputs,
                task_ds,
            **task_kwargs)
            print(results)
    def benchmark_all(self, metric: str, cache_model_segment : str, ds_segment: str, cache_model_idyom: str, ds_idyom: str, periode_frames: float, time_offset_frame: int) :
        model_outputs_segment = self.dataset_provider.get_ds(cache_model_segment)
        model_outputs_segment = model_outputs_segment.remove_columns(['Time']) # TODO: remove
        task_ds_segment = self.dataset_provider.get_ds(ds_segment)
        results = [
            segment_boundary_alignment(
                model_outputs_segment,
                task_ds_segment,
                periode_frames=periode_frames,
                time_offset_frame=time_offset_frame,
                repetition_mode=segment_inclusion,
                hiearchy_levels = ['lowercase', 'uppercase', 'functions']
            ) for segment_inclusion in ['all', 'only_first_occurence', 'change']
        ]
        results.append(
            repetition_decrease(
                model_outputs = model_outputs_segment,
                metadata_ds = task_ds_segment,
                periode_frames = periode_frames,
                time_offset_frame = time_offset_frame,
                normalize = True
            )
        )

        model_outputs_idyom = self.dataset_provider.get_ds(cache_model_idyom)
        # model_outputs_idyom = model_outputs_idyom.remove_columns(['Time']) # TODO: remove
        task_ds_idyom = self.dataset_provider.get_ds(ds_idyom)

        task_ds_idyom = task_ds_idyom.select_columns(['id', 'onset', f'idyom_{metric}']).rename_column(f'idyom_{metric}', 'idyom_val')
        results.append(
            idyom_corr(
                val_dataset = model_outputs_idyom,
                idyom_ds = task_ds_idyom,
                periode_frames = periode_frames,
                combine_inst = True,
                remove_outliers = False,
            )
        )
        # TODO: streamline returns
if __name__ == '__main__':
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
    CLI(BenchmarkSuite, as_positional=True, default_config_files=[str(config_path)])