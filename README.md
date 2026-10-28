# MEX: Musical Expectation Benchmark Suite

MEX is a Python package for benchmarking computational models of musical expectation.

Supported tasks currently include:

- Repetition decrease in information content
- Segment boundary detection
- IDyOM consistency
- Neural encoding of musical expectation in EEG

MEX also provides AudioIC and PA-AudioIC baselines and preprocessed datasets through the
[MEX dataset on Hugging Face](https://huggingface.co/datasets/MusicPerceptModeling/mex).

## Installation

```bash
git clone https://github.com/MusicPerceptModeling/mex.git
cd mex
pip install .
```

Optional dependencies:

```bash
pip install ".[baselines]"  # AudioIC and PA-AudioIC
pip install ".[extras]"
pip install ".[dev]"
pip install ".[demo]"
pip install ".[all]"
```

The baseline dependencies may require a separate PyTorch installation prior to installing. See the
[PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Usage

MEX provides a Python API and a command-line interface based on
[jsonargparse](https://jsonargparse.readthedocs.io).

```bash
python -m mex.cli --help
```

Default CLI settings are defined in [`config/config.yaml`](./config/config.yaml) and can
be overridden from the command line:

```bash
python -m mex.cli --dataset_provider.cache_dir=./cache
```

An API example is available in [`demo.ipynb`](./demo.ipynb).

## Workflows

Preprocess metadata (Typically skip running with preprocessed datasets from the hub):

```bash
python -m mex.cli --config config/preproc_salami.yaml
python -m mex.cli --config config/preproc_session.yaml
```

Compute baseline outputs:

```bash
python -m mex.cli --config config/baseline_audioic_session.yaml
python -m mex.cli --config config/baseline_pa_audioic_session.yaml
python -m mex.cli --config config/baseline_audioic_song.yaml
```

Run benchmarks:

```bash
python -m mex.cli --config config/benchmark_repetition_decrease_salami.yaml
python -m mex.cli --config config/benchmark_segment_boundary_salami.yaml
python -m mex.cli --config config/benchmark_idyom_consistency_session.yaml
python -m mex.cli --config config/benchmark_neural_encoding_song.yaml
```

Configuration files can be adapted for compatible datasets and model outputs.

## Contributing

Contributions of new tasks, datasets, metadata schemes, and baselines are welcome.
Install development dependencies with:

```bash
pip install ".[dev]"
```

### Adding a task

Implement [`Task`](./mex/benchmark.py) and its `_run_task` method. Tasks receive:

- `midm`: a Hugging Face `Dataset` containing an `id` column and frame-level model
  predictions such as information content or entropy
- `metadata`: a matching `id` column and task-specific annotations

Use `parse_metadata_scheme_fn` to validate metadata. See
[`RepetitionDecreaseTask`](./mex/task/repetition_decrease.py) for an example.

### Adding a dataset

Implement a [`MetaData`](./mex/benchmark.py) subclass when a dataset needs preprocessing
or a reusable metadata schema. Define `_preprocess`, `ds_name`, and
`parse_metadata_scheme`; shared schemas can be factored into a base class such as
[`SegmentMetaData`](./mex/metadata/segment.py).

### Adding precomputed datasets

Run preprocessing with `--dataset_provider.hf_upload=true` to write the dataset to
[`hf_datasets_mex`](./hf_datasets_mex/). Commit both the generated dataset and the
submodule pointer:

```bash
python -m mex.cli --config config/preproc_salami.yaml \
  --dataset_provider.hf_upload=true

git -C hf_datasets_mex add salami
git -C hf_datasets_mex commit -m "Update SALAMI dataset"
git add hf_datasets_mex
git commit -m "Update hf_datasets_mex submodule pointer"
```

Keep dataset assets and the corresponding MEX metadata implementation compatible.

## License

This project is licensed under the MIT license. See [`LICENSE`](./LICENSE).

Third-party code and precomputed assets may have additional license terms. See
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) and the
[MEX Hugging Face dataset](https://huggingface.co/datasets/MusicPerceptModeling/mex).
