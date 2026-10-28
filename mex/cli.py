from pathlib import Path
import re
from typing import List

from jsonargparse import CLI
from dataclasses import dataclass

import datasets

from .baseline import Baseline
from .benchmark import MetaData, Task
from .task import RepetitionDecreaseTask
from .task import SegmentBoundaryTask
from .task import IDyOMConsistencyTask
from .dataset_provider import DatasetProvider


def extract_numeric_stem_id(path: str) -> int:
    return int(re.search(r"\d+", Path(path).stem).group())

@dataclass
class BenchmarkSuite:
    dataset_provider: DatasetProvider
    def preprocess_metadata(self, metadata: MetaData):
        metadata.preprocess(self.dataset_provider)

    def baseline(self, model: Baseline, audio_ds : str | datasets.Dataset , cache_name : str, val: str = 'ic'):
        if isinstance(audio_ds, str):
            audio_ds = self.dataset_provider.get_ds(audio_ds)
        if val == 'ic':
            ic_ds = model.ic(audio_ds)
        elif val == 'entropy':
            ic_ds = model.entropy(audio_ds)
        self.dataset_provider.persist_ds(ic_ds, ds_name=cache_name)

    def benchmark(self, task: Task, midm: str | List[str], metadata: MetaData | str, task_kwargs: dict = {}, out_path: str = None, print_result: bool = True):
        if isinstance(midm, str):
            midm = self.dataset_provider.get_ds(midm, use_cache=False)
        else:
            midm_ = [self.dataset_provider.get_ds(midm[0],use_cache=False)]
            for cm in midm[1:]:
                midm_.append(self.dataset_provider.get_ds(cm, use_cache=False).remove_columns(['id'])) # TODO: remove time
            midm = datasets.concatenate_datasets(midm_, axis=1)
        if isinstance(metadata, str):
            metadata = self.dataset_provider.get_ds(metadata)
        elif isinstance(metadata, MetaData):
            metadata = metadata.load(dataset_provider=self.dataset_provider)
        # TODO: possibly put the task_kwargs in the constructor instead
        result = task.run_task(midm=midm, metadata=metadata, **task_kwargs)
        if out_path is not None:
            result.save(path=out_path)
        if print_result:
            print(result)


if __name__ == '__main__':
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
    CLI(BenchmarkSuite, as_positional=True, default_config_files=[str(config_path)])
