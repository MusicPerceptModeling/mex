
<!-- 
# TODO: show API and addtionally show other benchmarks.
 -->
# MEX: Musical Expectation Benchmark Suite

Installable package for benchmarking computational models of musical expectation on different tasks.

Current focus: an example benchmark that checks whether repetition lowers information content on SALAMI-style annotations. IDyOM consistency is also supported.

More benchmark tasks will be added later.

## Installation

PyTorch is required for baselines. Install it first: https://pytorch.org/get-started/locally/

```bash
# core
pip install .

# baselines
pip install ".[baselines]"

# extras
pip install ".[extras]"

# tests
pip install ".[tests]"

# demo
pip install ".[demo]"

# all optional groups
pip install ".[all]"
```

## Usage
MEX impliments an API for running benchmarks and a cli build with [jsonargparse](https://jsonargparse.readthedocs.io).

### Preprocess data
#### SALAMI
CLI (with config):

```bash
python -m mex.cli --config config/preproc_salami.yaml
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
	hf_root="muthissar/music_surprisal_benchmarks",
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

For a full runnable example, see [`demo.ipynb`](demo.ipynb).


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
	hf_root="muthissar/music_surprisal_benchmarks",
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
### Adding a task 
This requires implimenting a [`Task`](./mex/benchmark.py).
See the implimentation of [`RepetitionDecreaseTask`](./mex/task/repetition_decrease.py).
`Task`s impliment `_run_task`, which requires `midm` and `metadata` to follow a specific structure. 
`midm` usually contains at least two columns: 
- audio file `id`
- frame-based model predictions (e.g. *Information Content*)

`metadata` usually contains an `id` column for matching predictions and audio data and addtional data. To ensure that the provided metadata is compatible with the task `parse_metadata_scheme_fn: type[MetaData] | Callable[[Dataset], Dataset]` can be parsed to `Task`'s constructor. Parsing a `MetaData` `class` is pratical in cases where a dataset is shared between multiple tasks (e.g.,  [`SegmentMetaData`](./mex/metadata/segment.py) is used both in [`RepetitionDecreaseTask`](./mex/task/repetition_decrease.py) and [`SegmentBoundaryTask`](./mex/task/segment_boundary.py)).
### Adding a dataset to an existing task
To facilitate running the same task on different datasets, it is encouraged to contribute datasets in the form of an abstract implimentation of [`MetaData`](./mex/benchmark.py) (e.g., [`SegmentMetaData`](./mex/metadata/segment.py)) for shared parsing of metadata schemes which is implimented by another class that impliments the `_preprocess` method for preprocessing the dataset and `ds_name` for caching the preprocessed dataset (e.g., [`SalamiMetaData`](./mex/metadata/segment.py)).


Benchmark assets for tested datasets/baselines are provided as [`Dataset`s from the huggingface hub](https://huggingface.co/datasets/muthissar/music_surprisal_benchmarks).
<!--Scheme parsing of `Dataset`'s [`Features`](https://huggingface.co/docs/datasets/about_dataset_features) are used to ensure that the provided datasets can run the benchmark. This typically involves casting the `Dataset` to predefined `Features` (See, e.g., [`repetition_decrease`](mex/benchmark_repetition_test.py#L349-L361)).
-->
With the exception of the [`SessionMetaData`](./mex/metadata/symbolic.py) audio files for benchmarks are not provided.

## License

The project license has not been decided yet.
The [LICENSE](LICENSE) file is intentionally empty for now.

Code adapted from third-party projects (including `libfmp`) is acknowledged in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and remains under its original license terms.
