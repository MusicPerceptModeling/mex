from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, List
from warnings import warn
from datasets import Dataset
from datasets.features import Features, Value, Sequence
import pandas as pd

from ic_benchmark_suite.dataset_provider import DatasetProvider

@dataclass
class Result(ABC):
    def save(self, path: str):
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Result':
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

class PandasTestResult:
    def __init__(self, summary: pd.DataFrame):
        self.summary = summary
    def __str__(self) -> str:
        return self.summary.to_string(index=False)
    def save(self, path: str):
        self.summary.to_parquet(path)
    @classmethod
    def load(cls, path: str) -> 'PandasTestResult':
        summary = pd.read_parquet(path)
        return cls(summary=summary)


@dataclass
class MetaData(ABC):
    dataset_provider: DatasetProvider = field(default_factory=DatasetProvider())
    @abstractmethod
    def _preprocess(self) -> Dataset:
        pass
    @property
    @abstractmethod
    def ds_name(self) -> str:
        pass
    @property
    def cache_name(self) -> str:
        return self.dataset_provider.cache_dir + "/" + self.ds_name
    def preprocess(self):
        meta_data = self._preprocess()
        try:
            self.parse_metadata_scheme(meta_data)
        except Exception as e:
            warn(f"Could not parse metadata. Failed with {e}")
        self.dataset_provider.persist_ds(meta_data, ds_name=self.cache_name, hub_upload=False)
    @classmethod
    def parse_metadata_scheme(cls, dataset: Dataset):
        pass
    def load(self) -> Dataset:
        try:
            ds = self.dataset_provider.get_ds(self.cache_name)
            self.parse_metadata_scheme(ds)
            return ds
        except Exception as e:
            raise ValueError(f"Could not load metadata dataset {self.cache_name}. Error: {e}. Consider running preprocess first.")

class Task(ABC):
    def __init__(self, parse_metadata_scheme_fn: type[MetaData] | Callable[[Dataset], Dataset] = MetaData):
        self.parse_metadata_scheme = parse_metadata_scheme_fn.parse_metadata_scheme if isinstance(parse_metadata_scheme_fn, type) and issubclass(parse_metadata_scheme_fn, MetaData) else parse_metadata_scheme_fn    
    @classmethod
    def _midm_scheme(cls, labels: List[str]) -> Features:
        return Features(
            {
                "id": Value("int64"),
                **{label: Sequence(Value("float64")) for label in labels}
            }
        )
    def parse_midm_scheme(self, dataset: Dataset):
        assert "id" in dataset.column_names, "Dataset must contain 'id' column."
        midm_cols = self.midm_cols(dataset)
        if not midm_cols:
            raise ValueError("labels must not be empty")
        scheme = self._midm_scheme(midm_cols)
        
        # assert scheme == {feature: dataset.features[feature] for feature in scheme}, f"Dataset features {dataset.features} do not match required features {scheme} for midm."
        dataset.select_columns(list(scheme.keys())).cast(scheme)
        # return dataset

    def midm_cols(self, dataset : Dataset):
        # TODO: don't hard code anything else than id.
        # for now accept, since more complex schemes for midms are hard to concat
        midm_cols = [col for col in dataset.column_names if col not in ["id"]]
        return midm_cols

    @abstractmethod
    def _run_task(
        self,
        midm: Dataset,
        metadata: Dataset,
        **kwargs
    ) -> Result:
        pass
    def run_task(
        self,
        midm: Iterable | Dataset,
        # NOTE: alternatively, metadata could be part of initializer.
        metadata: Iterable | Dataset, 
        **kwargs
    ) -> Result:
        if isinstance(midm, Iterable) and not isinstance(midm, Dataset):
            midm = Dataset.from_list(midm)
        if isinstance(metadata, Iterable) and not isinstance(metadata, Dataset):
            metadata = Dataset.from_list(metadata)
        try:
            self.parse_midm_scheme(midm)
        except Exception as e:
            raise ValueError(f"Error occurred while parsing midm scheme. {e}")
        try:
            self.parse_metadata_scheme(metadata)
        except Exception as e:
            raise ValueError(f"Error occurred while parsing metadata scheme. {e}")

        return self._run_task(midm, metadata, **kwargs)
