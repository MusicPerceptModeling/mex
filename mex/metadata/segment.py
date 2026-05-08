from dataclasses import dataclass

import pandas as pd

from ..benchmark import MetaData


from datasets import Dataset, Features, List as FList, Sequence, Value


import re
from pathlib import Path
from typing import Dict, List

from ..dataset_provider import DatasetProvider


@dataclass
class SegmentMetaData(MetaData):
    @classmethod
    def parse_metadata_scheme(cls, dataset: Dataset):
        annotation_names = [col for col in dataset.column_names if col != "id"]
        scheme = cls._metadata_scheme(annotation_names)
        assert scheme == {feature: dataset.features[feature] for feature in scheme}, f"Dataset features {dataset.features} do not match required features {scheme} for SegmentMetaData. "

    @classmethod
    def _metadata_scheme(cls, annotation_names: List[str]) -> Features:
        return Features(
            {
                "id": Value("int64"),
                **{f"{name}": FList({
                    "anidx": Value("string"),
                    "end_sec": Sequence(Value("float64")),
                    "start_sec": Sequence(Value("float64")),
                    "type": Sequence(Value("string")),
                }) for name in annotation_names},
            }
        )


class SalamiMetaData(SegmentMetaData):
    def __init__(self, salami_root: Path = None,  n_proc: int | None = None):
        super().__init__()
        self.salami_root = salami_root
        self.n_proc = n_proc

    @classmethod
    def _parse_annotation_file(cls, path_str: str) -> dict:
        pattern = re.compile(r"^.*/(?P<id>\d+)/parsed/textfile(?P<aid>[^_/]+)_(?P<level>functions|uppercase|lowercase)\.txt$")
        match = pattern.match(path_str)
        if match is None:
            raise ValueError(f"Unexpected SALAMI annotation path format: {path_str}")

        raw_df = pd.read_csv(path_str, sep=r"\s+", header=None, names=["start_sec", "type"])
        raw_df["start_sec"] = pd.to_numeric(raw_df["start_sec"], errors="coerce")
        raw_df["type"] = raw_df["type"].astype(str)
        raw_df = raw_df.dropna(subset=["start_sec"]).reset_index(drop=True)

        seg_dict = {
            "anidx": str(match.group("aid")),
            "start_sec": raw_df["start_sec"].iloc[:-1].astype(float).tolist(),
            "end_sec": raw_df["start_sec"].iloc[1:].astype(float).tolist(),
            "type": raw_df["type"].iloc[:-1].astype(str).tolist(),
        }
        return {
            "id": int(match.group("id")),
            "level": match.group("level"),
            "segment": seg_dict,
        }

    @classmethod
    def _parsed_features(cls) -> Features:
        return Features(
            {
                "id": Value("int64"),
                "level": Value("string"),
                "segment": {
                    "anidx": Value("string"),
                    "start_sec": Sequence(Value("float64")),
                    "end_sec": Sequence(Value("float64")),
                    "type": Sequence(Value("string")),
                },
            }
        )

    @classmethod
    def read_annotations_hf(
        cls,
        folder: Path,
        cache_dir: str | None = None,
        remove_ids: set[int] = {},
        n_proc = None
    ) -> Dataset:
        """Read SALAMI annotations as a cached HF Dataset using parallel per-file parsing."""
        pattern = re.compile(r"^.*/(?P<id>\d+)/parsed/textfile[^_/]+_(functions|uppercase|lowercase)\.txt$")

        def iter_paths(folder: Path, keep_ids: set[int] | None):
            for p in (folder / "salami-data-public" / "annotations").rglob("textfile*_*.txt"):
                path_str = str(p)
                if keep_ids is not None:
                    match = pattern.match(path_str)
                    if match is None or int(match.group("id")) in remove_ids:
                        continue
                yield {"path": path_str}

        files_ds = Dataset.from_generator(
            iter_paths,
            gen_kwargs={"folder": folder, "keep_ids": remove_ids},
            cache_dir=cache_dir,
        ).sort("path")
        cache_file_name = None
        if cache_dir is not None:
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_file_name = str(cache_root / "salami_annotations_parsed.arrow")

        parsed = files_ds.map(
            cls._parse_annotation_file,
            num_proc=n_proc,
            load_from_cache_file=True,
            cache_file_name=cache_file_name,
            features=cls._parsed_features(),
            input_columns=["path"],
            remove_columns=["path"],
        )

        by_id: Dict[int, dict] = {}
        for ex in parsed:
            id_bucket = by_id.setdefault(int(ex["id"]), {"id": int(ex["id"])})
            segment_bucket = id_bucket.setdefault(str(ex["level"]), [])
            segment_bucket.append(ex["segment"])

        rows = [by_id[id_] for id_ in sorted(by_id)]
        return Dataset.from_list(rows, features=SegmentMetaData._metadata_scheme(["functions", "lowercase", "uppercase"]))

    @property
    def ds_name(self):
        return 'salami'

    def _preprocess(self, cache_dir: str = None) -> Dataset:
        assert self.salami_root is not None, "salami_root must be provided for preprocessing SalamiMetaData"
        return self.read_annotations_hf(
            folder=self.salami_root,
            cache_dir=cache_dir,
            n_proc=self.n_proc
        )
