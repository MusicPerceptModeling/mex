
<!--
# TODO: show API and addtionally show other benchmarks.
 -->
# MEX: Musical Expectation Benchmark Suite

Installable package for benchmarking computational models of musical expectation on different tasks.

Current focus: an example benchmark that checks whether repetition lowers information content on SALAMI-style annotations. IDyOM consistency is also supported.

More benchmark tasks will be added later.

## Installation
<!-- TODO: update when the repo names changes --->
### Getting
```
git clone https://github.com/PerceptualMusicModeling/mex.git
cd music_surprisal_benchmarks
```
### Installing
```bash
# core
pip install .

# baselines
# First install PyTorch: https://pytorch.org/get-started/locally/
pip install ".[baselines]"

# extras
pip install ".[extras]"

# development
pip install ".[dev]"

# demo
pip install ".[demo]"

# all optional groups
pip install ".[all]"
```

## Usage
MEX impliments an API for running benchmarks and a cli build with [jsonargparse](https://jsonargparse.readthedocs.io).

For documentation on the API see [`./demo.ipynb`](./demo.ipynb).
For documentation on the cli, run 
```bash
python -m mex.cli --help
```
The cli stores default application parameters in [`./config/config.yaml`](./config/config.yaml), that can be overwritten by passing command line parameters, e.g.
 ```bash
python -m mex.cli --dataset_provider.cache_dir=./cache --help
```

MEX stores Benchmark assets for standard datasets and baselines as [`Dataset`](https://huggingface.co/docs/datasets/v4.8.4/en/package_reference/main_classes#datasets.Dataset)s in the huggingface hub repository [https://huggingface.co/datasets/PerceptualMusicModeling/mex](https://huggingface.co/datasets/PerceptualMusicModeling/mex).
<!--Scheme parsing of `Dataset`'s [`Features`](https://huggingface.co/docs/datasets/about_dataset_features) are used to ensure that the provided datasets can run the benchmark. This typically involves casting the `Dataset` to predefined `Features` (See, e.g., [`repetition_decrease`](mex/benchmark_repetition_test.py#L349-L361)).
-->
With the exception of the [`SessionMetaData`](./mex/metadata/symbolic.py) and [`NHSSMetaData`]() audio files for benchmarks are not provided.


### Preprocess metadata
(Skip if running with preprocessed datasets from the hub)
#### SALAMI
CLI (with config):

```bash
python -m mex.cli --config config/preproc_salami.yaml
```
### Calculate MIDMs
(skip if using own model, or running with precomputed baseline values from the hub)

CLI (with config):

```bash
python -m mex.cli --config config/baseline_audioic_session.yaml
```
### Run task
#### Reduction in MIDM with repetition
This repetition benchmark can be run on any compatible dataset; SALAMI is the example shown here.

CLI (with config):

```bash
python -m mex.cli --config config/benchmark_repetition_decrease_salami.yaml
```


<!--
API:

```python
import os
from mex import repetition_decrease
from mex.dataset_provider import DatasetProvider

dataset_provider = DatasetProvider(
	cache_dir="./cache",
	hf_root="PerceptualMusicModeling/mex",
	hf_token=os.environ.get("HF_ACCESS_TOKEN"),
)

# Get from hf_hub. For instance, for providing precomputed cache files where audio cannot be shared.
model_outputs = dataset_provider.get_ds("audioic_salami_ic")
# Get from local disk
metadata_ds = dataset_provider.get_ds("./cache/datasets/salami_metadata_ds")

results = repetition_decrease(
    model_outputs=model_outputs,
    metadata_ds=metadata_ds,
    periode_frames = 0.0928798185941,
    time_offset_frame =  1 # for IC 1 for entropy 0
)

``` -->


<!-- ## Baselines

The repository currently provides two baseline models:

- `mex.baseline.audioic.AudioIC`
- `mex.baseline.pa_audioic.PAAudioIC`

Minimal API usage:

```python
import os
from mex.dataset_provider import DatasetProvider
from mex.baseline.audioic import AudioIC
from mex.baseline.pa_audioic import PAAudioIC

dp = DatasetProvider(
	cache_dir="./cache",
	hf_root="PerceptualMusicModeling/mex",
	hf_token=os.environ.get("HF_ACCESS_TOKEN")
)

audio_ds = dp.get_ds("cache/datasets/session_ds")

# Information content
ic_ds = AudioIC(device="cuda").ic(audio_ds)

# Entropy
entr_ds = PAAudioIC(device="cuda", noise_levels=[0.0], monte_carlo_samples=16).entr(audio_ds)
```

Output columns:

- `AudioIC`: `ic`
- `PAAudioIC` IC: `ic_<noise_level>`
- `PAAudioIC` entropy: `entr_<noise_level>` -->

## Contributing
We warmly welcome contributions to MEX in the form of new tasks or datasets for existing tasks.

Ensure that mex was installed with dev dependencies: `pip install ".[dev]"`.

### Adding a task
This requires implimenting a [`Task`](./mex/benchmark.py).
See the implimentation of [`RepetitionDecreaseTask`](./mex/task/repetition_decrease.py).
`Task`s impliment `_run_task`, which requires `midm` and `metadata` to follow a specific structure.
`midm` usually contains at least two columns:
- audio file `id`
- frame-based model predictions (e.g. *Information Content*)

`metadata` usually contains an `id` column for matching predictions and audio data and addtional data. To ensure that the provided metadata is compatible with the task `parse_metadata_scheme_fn: type[MetaData] | Callable[[Dataset], Dataset]` can be passed to `Task`'s constructor. Passing a `MetaData` `class` (see below) is pratical in cases where a dataset is shared between multiple tasks (e.g.,  [`SegmentMetaData`](./mex/metadata/segment.py) is used both in [`RepetitionDecreaseTask`](./mex/task/repetition_decrease.py) and [`SegmentBoundaryTask`](./mex/task/segment_boundary.py)).
### Adding a dataset to an existing task
#### Adding a MetaData
To facilitate running the same task on different datasets, it is encouraged to contribute datasets in the form of an abstract implimentation of [`MetaData`](./mex/benchmark.py) (e.g., [`SegmentMetaData`](./mex/metadata/segment.py)) for shared parsing of metadata schemes which is implimented by another class that impliments the `_preprocess` method for preprocessing the dataset and `ds_name` for caching the preprocessed dataset (e.g., [`SalamiMetaData`](./mex/metadata/segment.py)).


#### Adding precomputed cache to the hub
Running `preprocess_metadata` with `--dataset_provider.hf_upload=true` writes the `MetaData` assets to the hf datasets git submodule [`hf_datasets_mex`](./hf_datasets_mex/). When updating `MetaData` implimentations it is recommended to commit both [`./hf_datasets_mex`](./hf_datasets_mex/) and [`.`]('./') to keep versions compatible, e.g.:  
```
python -m mex.cli --dataset_provider.hf_upload=true --config config/preproc_salami.yaml

git -C hf_datasets_mex add salami
git -C hf_datasets_mex commit -m "Updated SALAMI dataset"

git add hf_datasets_mex
git commit -m "Updated hf_datasets_mex submodule pointer"

```

## License

This project is licensed under the terms of the MIT license.

Code adapted from third-party projects (including `libfmp`) is acknowledged in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and remains under its original license terms.

Additional licenses apply to precomputed assets, please refer to [https://huggingface.co/datasets/PerceptualMusicModeling/mex](https://huggingface.co/datasets/PerceptualMusicModeling/mex).
