import math
import re
from typing import Dict, List, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from datasets import Dataset, Features, Sequence, Value


def _salami_id_extractor(path: Path) -> int:
    match = re.match(r"^\D*(?P<id>\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Could not parse SALAMI id from filename: {path}")
    return int(match.group("id"))


def _parse_salami_annotation_file(path_str: str) -> dict:
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


def _salami_parsed_features() -> Features:
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


def read_salami_annotations_hf(
    folder: Path,
    cache_dir: str | None = None,
    keep_ids: set[int] | None = None,
    n_proc = None
) -> Dataset:
    """Read SALAMI annotations as a cached HF Dataset using parallel per-file parsing."""
    pattern = re.compile(r"^.*/(?P<id>\d+)/parsed/textfile[^_/]+_(functions|uppercase|lowercase)\.txt$")

    def iter_paths(folder: Path, keep_ids: set[int] | None):
        for p in (folder / "salami-data-public" / "annotations").rglob("textfile*_*.txt"):
            path_str = str(p)
            if keep_ids is not None:
                match = pattern.match(path_str)
                if match is None or int(match.group("id")) not in keep_ids:
                    continue
            yield {"path": path_str}

    files_ds = Dataset.from_generator(
        iter_paths,
        gen_kwargs={"folder": folder, "keep_ids": keep_ids},
        cache_dir=cache_dir,
    ).sort("path")
    cache_file_name = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file_name = str(cache_root / "salami_annotations_parsed.arrow")

    parsed = files_ds.map(
        _parse_salami_annotation_file,
        num_proc=n_proc,
        load_from_cache_file=True,
        cache_file_name=cache_file_name,
        features=_salami_parsed_features(),
        input_columns=["path"],
        remove_columns=["path"],
    )

    by_id: Dict[int, dict] = {}
    for ex in parsed:
        id_bucket = by_id.setdefault(int(ex["id"]), {"id": int(ex["id"])})
        segment_bucket = id_bucket.setdefault(str(ex["level"]), [])
        segment_bucket.append(ex["segment"])

    rows = [by_id[id_] for id_ in sorted(by_id)]
    return Dataset.from_list(rows)


def merge_nll_with_salami_metadata(nll_dataset, salami_metadata_df):
    """Join nll rows with SALAMI metadata on id and return row dictionaries."""
    nll_df = nll_dataset.to_pandas()
    merged = nll_df.merge(salami_metadata_df, on="id", how="inner")
    return merged.to_dict(orient="records")


def extract_segment_df_from_nll_dataset(
    nll_dataset,
    segment_names: List[str],
    period_music_2_latent: float,
):
    """
    Build a segment-level table equivalent to the logic used in eval.py.

    Expected row fields in nll_dataset:
    - id, nll, is_silence
    - optional: noise_level
    - each segment name in segment_names with value dict or list[dict]
      containing start_sec, end_sec, type, optional anidx
    """
    out_rows = []

    for ex in nll_dataset:
        nlls = np.asarray(ex["nll"])
        if nlls.ndim == 1:
            nlls = nlls[:, None]
        elif nlls.ndim != 2:
            raise ValueError(f"nll should be 1D or 2D, got shape={nlls.shape}")
        nlls = nlls.copy()
        if "is_silence" in ex:
            # raise ValueError("Missing required field 'is_silence' in example")
            is_silence = np.asarray(ex["is_silence"], dtype=bool)
            nlls[is_silence] = np.nan

        for seg_name in segment_names:
            if seg_name not in ex:
                continue

            seg = ex[seg_name]
            if isinstance(seg, Mapping):
                segs = [seg]
            elif isinstance(seg, list) or isinstance(seg, np.ndarray):
                segs = seg
            else:
                raise ValueError(f"Expected dict or list for {seg_name}, got {type(seg)}")

            for segment in segs:
                end_secs = np.asarray(segment["end_sec"])
                start_secs = np.asarray(segment["start_sec"])
                types = np.asarray(segment["type"])

                frames = math.ceil(np.nanmax(end_secs) / period_music_2_latent)
                pad_len = max(frames - len(nlls), 0)
                if pad_len > 0:
                    pad = np.empty((pad_len, nlls.shape[1]))
                    pad.fill(np.nan)
                    nlls_padded = np.concatenate([nlls, pad], axis=0)
                else:
                    nlls_padded = nlls

                type_counts = {}
                for i, (start, end, type_) in enumerate(zip(start_secs, end_secs, types)):
                    start_frame = round(start / period_music_2_latent)
                    end_frame = round(end / period_music_2_latent)
                    nll_window = nlls_padded[start_frame:end_frame]

                    if len(nll_window) > 0:
                        mean_nll = np.nanmean(nll_window, axis=0).tolist()
                        min_nll = np.nanmin(nll_window, axis=0).tolist()
                        max_nll = np.nanmax(nll_window, axis=0).tolist()
                    else:
                        mean_nll = [np.nan] * nlls_padded.shape[1]
                        min_nll = [np.nan] * nlls_padded.shape[1]
                        max_nll = [np.nan] * nlls_padded.shape[1]

                    nrep = type_counts.setdefault(type_, 0)
                    type_counts[type_] += 1

                    out_rows.append(
                        {
                            "id": ex["id"],
                            "seg_name": seg_name,
                            "anidx": segment.get("anidx", "-1"),
                            "noise_level": ex.get("noise_level", -1),
                            "type": type_,
                            "nrep": float(nrep),
                            "nseg": i,
                            "mean_nll": mean_nll,
                            "min_nll": min_nll,
                            "max_nll": max_nll,
                        }
                    )

    segment_df = pd.DataFrame(out_rows)

    if segment_df.empty:
        return segment_df

    segment_df = segment_df.explode(["mean_nll", "min_nll", "max_nll", "noise_level"]).convert_dtypes()
    segment_df.loc[segment_df["noise_level"].isna(), "noise_level"] = -1
    return segment_df


def run_repetition_test(segment_df):
    """
    Reproduce the notebook test per segment name:
    - group by (id, type) within each segment name
    - sort by nrep
    - compute diff(mean_nll)
    - one-sided t-test: mean(diff_val) < 0
    
    Returns:
        dict mapping seg_name -> (summary Series, residual DataFrame)
    """
    df = segment_df.copy()

    required_cols = {"id", "type", "nrep", "mean_nll", "seg_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"segment_df is missing required columns: {sorted(missing)}")

    df["mean_nll"] = pd.to_numeric(df["mean_nll"], errors="coerce")

    # TODO: possible do stronger statistical test, noting that there are multiple annotations per file
    # and that the observations per annotation might not be independent.
    results = {}
    for seg_name in df["seg_name"].unique():
        seg_df = df[df["seg_name"] == seg_name].copy()
        seg_df = seg_df.sort_values(["id", "type", "anidx", "nrep"])
        seg_df["diff_val"] = seg_df.groupby(["id", "type", "anidx"])["mean_nll"].diff()
        res = seg_df.dropna(subset=["diff_val"])
        stat_res = scipy.stats.ttest_1samp(res.diff_val, popmean=0.0, alternative="less")
        summary = pd.Series(
            [
                res.diff_val.mean(),
                stat_res.statistic,
                stat_res.pvalue,
                stat_res.df,
                *stat_res.confidence_interval(0.95),
            ],
            index=["mean", "t_stat", "p_value", "df", "conf_low", "conf_high"],
        )
        results[seg_name] = (summary, res)

    return results

def _load_nll_csvs_as_rows(csv_paths) -> list:
    rows = []
    for p in csv_paths:
        p = Path(p)
        id_ = _salami_id_extractor(p)
        df = pd.read_csv(p)
        row = {"id": id_}
        for col in df.columns:
            row[col] = df[col].tolist()
        rows.append(row)
    return rows


def repetition_decrease(nll_dataset, salami_root, periode_frames, n_proc=None, segment_names=['uppercase'], cache_dir=None):
    real_salami_metadata_df = read_salami_annotations_hf(salami_root, cache_dir=cache_dir, n_proc=n_proc).to_pandas()
    merged_rows = merge_nll_with_salami_metadata(nll_dataset, real_salami_metadata_df)
    segment_df = extract_segment_df_from_nll_dataset(
        nll_dataset=merged_rows,
        segment_names=segment_names,
        period_music_2_latent=periode_frames,
    )
    results = run_repetition_test(segment_df)
    return results

if __name__ == "__main__":
    # nll_csv_dir = Path('/home/mathias/devel/python3/audioic/salami')
    nll_csv_dir = Path('/home/mathias/devel/python3/pa-audioic/salami')
    csv_paths = sorted(nll_csv_dir.glob('*.csv'), key=_salami_id_extractor)
    nll_dataset = Dataset.from_list(_load_nll_csvs_as_rows(csv_paths))
    # nll_dataset = nll_dataset.rename_column('IC', 'nll')
    nll_dataset = nll_dataset.rename_column('IC_0.0', 'nll')

    n_proc = 8
    segment_names = ["lowercase", "uppercase", "functions"]
    salami_root = "/opt/datasets/fs/salami"
    cache_dir = "/home/mathias/devel/python3/audio-it/ic_benchmark_suite/cache"
    # TODO, if not provided, we could assume something.
    periode_frames = 0.093
    salami_root = Path(salami_root)
    results = repetition_decrease(nll_dataset, salami_root, periode_frames, n_proc, segment_names, cache_dir)

    summary_lowercase = results["lowercase"][0]
    expected_lowercase = {
        "mean": -9.292529e-01,
        "t_stat": -2.358669e+01,
        "p_value": 9.069782e-123,
        "df": 6.254000e+04,
        "conf_high": -8.644491e-01,
    }

    assert np.isclose(summary_lowercase["mean"], expected_lowercase["mean"])
    assert np.isclose(summary_lowercase["t_stat"], expected_lowercase["t_stat"])
    assert np.isclose(summary_lowercase["p_value"], expected_lowercase["p_value"])
    assert np.isclose(summary_lowercase["df"], expected_lowercase["df"])
    assert np.isneginf(summary_lowercase["conf_low"])
    assert np.isclose(summary_lowercase["conf_high"], expected_lowercase["conf_high"])
    
    summary_uppercase = results["uppercase"][0]
    expected_uppercase = {
        "mean": -2.622573e+00,
        "t_stat": -1.981856e+01,
        "p_value": 1.919504e-86,
        "df": 1.299000e+04,
        "conf_high": -2.404895e+00,
    }

    assert np.isclose(summary_uppercase["mean"], expected_uppercase["mean"])
    assert np.isclose(summary_uppercase["t_stat"], expected_uppercase["t_stat"])
    assert np.isclose(summary_uppercase["p_value"], expected_uppercase["p_value"])
    assert np.isclose(summary_uppercase["df"], expected_uppercase["df"])
    assert np.isneginf(summary_uppercase["conf_low"])
    assert np.isclose(summary_uppercase["conf_high"], expected_uppercase["conf_high"])