from dataclasses import dataclass

import pandas as pd

from ..benchmark import MetaData


from datasets import Dataset, Features, List as FList, Sequence, Value


import re
from pathlib import Path
from typing import Dict, List

from ..dataset_provider import DatasetProvider


@dataclass
class NeuralEncodingMetaData(MetaData):
    @classmethod
    def parse_metadata_scheme(cls, dataset: Dataset):
        annotation_names = [col for col in dataset.column_names if col != "id"]
        print(annotation_names)
        #scheme = cls._metadata_scheme(annotation_names)
        #assert scheme == {feature: dataset.features[feature] for feature in scheme}, f"Dataset features {dataset.features} do not match required features {scheme} for SegmentMetaData. "

    #@classmethod
    #def _metadata_scheme(cls, annotation_names: List[str]) -> Features:
    #    return Features(
    #        {
    #            "id": Value("int64"),
    #            **{f"{name}": FList({
    #                "anidx": Value("string"),
    #                "end_sec": Sequence(Value("float64")),
    #                "start_sec": Sequence(Value("float64")),
    #                "type": Sequence(Value("string")),
    #            }) for name in annotation_names},
    #        }
    #    )

class SongMetaData(NeuralEncodingMetaData):
    def __init__(self, song_root: Path = None,  n_proc: int | None = None):
        super().__init__()
        self.song_root = song_root
        self.n_proc = n_proc

    @property
    def ds_name(self):
        return 'song'

    def _preprocess(self, cache_dir: str = None) -> Dataset:
        assert self.song_root is not None, "song_root must be provided for preprocessing SongMetaData"
        return self.read_annotations_hf(
            folder=self.song_root,
            cache_dir=cache_dir,
            n_proc=self.n_proc
        )
