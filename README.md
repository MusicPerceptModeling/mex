# ic_benchmark_suite

Installable package for IC benchmark tasks.

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

### Benchmark repetition
This repetition benchmark can be run on any compatible dataset; SALAMI is the example shown here.

CLI (with config):

```bash
python -m ic_benchmark_suite.cli --config config/benchmark_repetition.yaml benchmark
```

API:

```python
import os
from ic_benchmark_suite.dataset_provider import DatasetProvider

dataset_provider = DatasetProvider(
	cache_dir="./cache",
	hf_root="muthissar/music_surprisal_benchmarks",
	hf_token=os.environ.get("HF_ACCESS_TOKEN"),
)
model_output = dataset_provider.get_ds("audioic_salami_ic")
```

For a full runnable example, see [`demo.ipynb`](demo.ipynb).

### Benchmark ...

## Baselines

The repository currently provides two baseline models:

- `ic_benchmark_suite.baseline.audioic.AudioIC`
- `ic_benchmark_suite.baseline.pa_audioic.PAAudioIC`

Minimal API usage:

```python
import os
from ic_benchmark_suite.dataset_provider import DatasetProvider
from ic_benchmark_suite.baseline.audioic import AudioIC
from ic_benchmark_suite.baseline.pa_audioic import PAAudioIC

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
- `PAAudioIC` entropy: `entr_<noise_level>`

## Contributing
We warmly welcome adding benchmarks. 
This, usually, requires implimenting a function that takes two [Huggingface `Dataset`](github.com/huggingface/datasets/) arguments:
- `model_outputs`,
- `metadata_ds`

See the current benchmark entry point: [`repetition_decrease`](ic_benchmark_suite/benchmark_repetition_test.py#L341).

`model_outputs` usually contains at least two columns: 
- audio file `id`
- frame-based model predictions (e.g. *Information Content*)

`metadata_ds` usually contains an `id` column, for matching predictions and audio data.

Benchmark assets for tested datasets/baselines are provided as [`Dataset`s from the huggingface hub](https://huggingface.co/datasets/muthissar/music_surprisal_benchmarks).

Scheme parsing of `Dataset`'s [`Features`](https://huggingface.co/docs/datasets/about_dataset_features) are used to ensure that the provided datasets can run the benchmark. This typically envolves casting the `Dataset` to predefined `Features` (See, e.g., [`repetition_decrease`](ic_benchmark_suite/benchmark_repetition_test.py#L349-L361)).

With the exception of the [`idyom_consistency`](ic_benchmark_suite/benchmark_idyom_corr_test.py#L148) audio files for benchmarks are not provided.