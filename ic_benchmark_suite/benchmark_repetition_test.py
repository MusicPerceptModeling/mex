import math
import re
from typing import Dict, List, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from datasets import Dataset


def _to_pandas(segment_df):
    """Accept either a Hugging Face Dataset or a pandas DataFrame."""
    if isinstance(segment_df, pd.DataFrame):
        return segment_df.copy()
    if hasattr(segment_df, "to_pandas"):
        return segment_df.to_pandas()
    return pd.DataFrame(segment_df)


def _to_rows(dataset_or_rows):
    """Accept HF Dataset, pandas DataFrame, list[dict], or iterable[dict]."""
    if isinstance(dataset_or_rows, pd.DataFrame):
        return dataset_or_rows.to_dict(orient="records")
    if hasattr(dataset_or_rows, "to_pandas"):
        return dataset_or_rows.to_pandas().to_dict(orient="records")
    if isinstance(dataset_or_rows, list):
        return dataset_or_rows
    return list(dataset_or_rows)


def _salami_id_extractor(path: Path) -> int:
    match = re.match(r"^\D*(?P<id>\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Could not parse SALAMI id from filename: {path}")
    return int(match.group("id"))


def _pick_annotation_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = [str(c) for c in df.columns]
    cols_lower = [c.lower() for c in cols]

    def first_match(candidates: List[str], default_idx: int) -> str:
        for cand in candidates:
            for i, c in enumerate(cols_lower):
                if cand in c:
                    return cols[i]
        return cols[min(default_idx, len(cols) - 1)]

    start_col = first_match(["start_sec", "start", "onset"], 0)
    end_col = first_match(["end_sec", "end", "offset"], 1)
    type_col = first_match(["type", "label", "segment", "function"], 2)
    return start_col, end_col, type_col


def _annotation_table_to_segment_dict(annotation_df: pd.DataFrame, anidx: int) -> dict:
    if annotation_df.empty:
        return {"anidx": anidx, "start_sec": [], "end_sec": [], "type": []}

    start_col, end_col, type_col = _pick_annotation_columns(annotation_df)

    start_vals = pd.to_numeric(annotation_df[start_col], errors="coerce")
    end_vals = pd.to_numeric(annotation_df[end_col], errors="coerce")
    type_vals = annotation_df[type_col].astype(str)
    keep = start_vals.notna() & end_vals.notna()

    return {
        "anidx": int(anidx),
        "start_sec": start_vals[keep].astype(float).tolist(),
        "end_sec": end_vals[keep].astype(float).tolist(),
        "type": type_vals[keep].tolist(),
    }


def _read_salami_annotations(folder: Path) -> pd.DataFrame:
    annotations_root = folder / "salami-data-public" / "annotations"
    pattern = re.compile(
        r"^.*/(?P<id>\d+)/parsed/textfile(?P<aid>\d+)_(?P<level>functions|uppercase|lowercase)\.txt$"
    )

    per_id: Dict[int, dict] = {}
    for path in annotations_root.rglob("textfile*_*.txt"):
        match = pattern.match(str(path))
        if match is None:
            continue
        id_ = int(match.group("id"))
        aid = int(match.group("aid"))
        level = match.group("level")

        # SALAMI parsed files are boundary lists (start_sec, type) without headers.
        # Build segment intervals by shifting start times to create end_sec.
        raw_df = pd.read_csv(path, sep=r"\s+", header=None, names=["start_sec", "type"])
        raw_df["start_sec"] = pd.to_numeric(raw_df["start_sec"], errors="coerce")
        raw_df["type"] = raw_df["type"].astype(str)
        raw_df = raw_df.dropna(subset=["start_sec"]).reset_index(drop=True)

        if raw_df.empty or len(raw_df) < 2:
            continue

        seg_df = pd.DataFrame(
            {
                "start_sec": raw_df["start_sec"].iloc[:-1].to_numpy(),
                "end_sec": raw_df["start_sec"].iloc[1:].to_numpy(),
                "type": raw_df["type"].iloc[:-1].to_numpy(),
            }
        )
        seg_dict = _annotation_table_to_segment_dict(seg_df, anidx=aid)

        id_bucket = per_id.setdefault(id_, {})
        id_bucket.setdefault(level, []).append(seg_dict)

    if not per_id:
        return pd.DataFrame(columns=["id", "functions", "uppercase", "lowercase"])

    anno_df = pd.DataFrame.from_dict(per_id, orient="index")
    anno_df.index.name = "id"
    anno_df = anno_df.reset_index()
    return anno_df


def load_salami_metadata_for_benchmark(
    salami_dir: str = "/opt/datasets/fs/salami",
    include_buggy: bool = False,
):
    """
    Minimal SALAMI metadata extraction adapted from eval.py.

    Returns a pandas DataFrame with at least:
    - id
    - part
    - annotation fields from SALAMI metadata (e.g. lowercase/uppercase/functions)
    """
    buggy_idx = [1327]

    salami_root = Path(salami_dir)
    audio_paths = list((salami_root / "salami_audios").glob("*.mp3"))

    if include_buggy:
        keep_ids = {_salami_id_extractor(p) for p in audio_paths}
    else:
        keep_ids = {
            _salami_id_extractor(p)
            for p in audio_paths
            if _salami_id_extractor(p) not in set(buggy_idx)
        }

    metadata_csv = salami_root / "salami-data-public" / "metadata" / "metadata.csv"
    metadata_df = pd.read_csv(metadata_csv, index_col="SONG_ID")
    metadata_df = metadata_df.loc[sorted(set(metadata_df.index).intersection(keep_ids))].copy()
    metadata_df.index.name = "id"
    metadata_df = metadata_df.reset_index()

    annotations_df = _read_salami_annotations(salami_root)
    merged = metadata_df.merge(annotations_df, on="id", how="left")

    n = len(merged)
    n_train = int(n * 10 / 12)
    n_valid = int(n * 1 / 12)
    order = np.random.default_rng(0).permutation(np.arange(n))
    part = np.array(["test"] * n, dtype=object)
    part[order[:n_train]] = "train"
    part[order[n_train : n_train + n_valid]] = "valid"
    merged["part"] = part

    return merged.sort_values("id").reset_index(drop=True)


def merge_nll_with_salami_metadata(nll_dataset, salami_metadata_df):
    """Join nll rows with SALAMI metadata on id and return row dictionaries."""
    nll_df = _to_pandas(nll_dataset)
    merged = nll_df.merge(salami_metadata_df, on="id", how="inner")
    return merged.to_dict(orient="records")


def check_nll_dataset_compatibility(nll_dataset, salami_metadata_df=None) -> dict:
    """Validate schema compatibility and optional id overlap with SALAMI metadata."""
    rows = _to_rows(nll_dataset)
    required = {"id", "nll", "is_silence"}
    errors = []

    for i, ex in enumerate(rows):
        missing = required - set(ex.keys())
        if missing:
            errors.append(f"row[{i}] missing keys: {sorted(missing)}")
            continue

        nll_len = len(ex["nll"])
        silence_len = len(ex["is_silence"])
        if nll_len != silence_len:
            errors.append(
                f"row[{i}] has len(nll)={nll_len} but len(is_silence)={silence_len}"
            )

    result = {
        "num_rows": len(rows),
        "schema_ok": len(errors) == 0,
        "errors": errors,
    }

    if salami_metadata_df is not None:
        nll_ids = {int(x["id"]) for x in rows if "id" in x}
        salami_ids = set(pd.to_numeric(salami_metadata_df["id"], errors="coerce").dropna().astype(int))
        overlap = sorted(nll_ids.intersection(salami_ids))
        result["id_overlap_count"] = len(overlap)
        result["id_overlap_sample"] = overlap[:10]
        result["id_compatible"] = len(overlap) > 0

    return result


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
    rows = _to_rows(nll_dataset)
    out_rows = []

    for ex in rows:
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
            elif isinstance(seg, list):
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
                            "anidx": segment.get("anidx", -1),
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


def diff(x: pd.DataFrame) -> pd.DataFrame:
    """Notebook-equivalent differencing across repetition index."""
    sorted_ = x.sort_values("nrep")
    return sorted_.assign(diff_val=sorted_.mean_nll.diff())


def run_repetition_test(segment_df):
    """
    Reproduce the notebook test:
    - group by (id, type)
    - sort by nrep
    - compute diff(mean_nll)
    - one-sided t-test: mean(diff_val) < 0
    """
    df = _to_pandas(segment_df)

    required_cols = {"id", "type", "nrep", "mean_nll"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"segment_df is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["mean_nll"] = pd.to_numeric(df["mean_nll"], errors="coerce")

    res = (
        df.groupby(["id", "type"])
        .apply(diff)
        .reset_index(drop=True)
        .dropna(subset=["diff_val"])
    )

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

    return summary, res


def _build_mock_nll_dataset(ids=None):
    """Small mock NLL dataset with per-frame IC and silence mask."""
    if ids is None:
        ids = [1001, 1002]
    if len(ids) < 2:
        raise ValueError("ids should contain at least 2 values")

    return [
        {
            "id": int(ids[0]),
            "noise_level": [0.0],
            "nll": [-4.0, -3.7, -4.3, -3.9, -4.5, -4.0, -4.6, -4.1],
            "is_silence": [False, False, False, False, False, False, False, False],
        },
        {
            "id": int(ids[1]),
            "noise_level": [0.0],
            "nll": [-5.1, -4.9, -5.4, -5.0, -5.7, -5.1, -5.8, -5.2],
            "is_silence": [False, False, False, False, False, False, False, False],
        },
    ]


def _build_mock_nll_dataset_from_salami_spans(
    period_music_2_latent: float = 0.093,
):
    """
    Build two hardcoded mock rows using provided SALAMI ids and time spans:
    - id 10: 0.0 to 314.269342403 sec
    - id 100: 0.0 to 143.531587301 sec
    """
    end_time_by_id = {
        10: 314.269342403,
        100: 143.531587301,
    }

    rows = []
    for sid in [10, 100]:
        end_sec = end_time_by_id[sid]
        n_frames = int(math.ceil(end_sec / period_music_2_latent))

        # Smooth deterministic signal with slight id-specific offset.
        t = np.arange(n_frames, dtype=float) * period_music_2_latent
        phase = 0.3 if sid == 10 else 0.8
        nll = -4.5 + 0.25 * np.sin(0.08 * t + phase)
        if sid == 100:
            nll = nll - 0.35

        rows.append(
            {
                "id": sid,
                "noise_level": [0.0],
                "nll": nll.tolist(),
                "is_silence": [False] * n_frames,
            }
        )

    return rows


def _build_mock_salami_metadata_df():
    """Small mock SALAMI-style metadata with segment labels."""
    return pd.DataFrame(
        [
            {
                "id": 1001,
                "part": "valid",
                "lowercase": [
                    {
                        "anidx": 0,
                        "start_sec": [0.0, 0.186, 0.372, 0.558],
                        "end_sec": [0.186, 0.372, 0.558, 0.744],
                        "type": ["a", "b", "a", "b"],
                    }
                ],
            },
            {
                "id": 1002,
                "part": "valid",
                "lowercase": [
                    {
                        "anidx": 1,
                        "start_sec": [0.0, 0.186, 0.372, 0.558],
                        "end_sec": [0.186, 0.372, 0.558, 0.744],
                        "type": ["a", "b", "a", "b"],
                    }
                ],
            },
        ]
    )
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


if __name__ == "__main__":
    # Demo pipeline with mock data:
    # 1) nll dataset
    # 2) SALAMI metadata merge
    # 3) segment extraction
    # 4) repetition hypothesis test
    real_salami_metadata_df = load_salami_metadata_for_benchmark()
    # dataset  = Dataset.from_csv(['/home/mathias/devel/python3/audioic/salami/10.csv', '/home/mathias/devel/python3/audioic/salami/11.csv'])
    salami_csv_dir = Path('/home/mathias/devel/python3/audioic/salami')
    csv_paths = sorted(salami_csv_dir.glob('*.csv'), key=_salami_id_extractor)
    mock_nll_dataset = Dataset.from_list(_load_nll_csvs_as_rows(csv_paths))
    mock_nll_dataset = mock_nll_dataset.rename_column('IC', 'nll')
    # mock_nll_dataset = _build_mock_nll_dataset_from_salami_spans(period_music_2_latent=0.093)
    compatibility = check_nll_dataset_compatibility(mock_nll_dataset, real_salami_metadata_df)
    print("NLL dataset compatibility:")
    print(compatibility)

    merged_rows = merge_nll_with_salami_metadata(mock_nll_dataset, real_salami_metadata_df)
    segment_df = extract_segment_df_from_nll_dataset(
        nll_dataset=merged_rows,
        segment_names=["lowercase"],
        period_music_2_latent=0.093,
    )
    summary, diff_df = run_repetition_test(segment_df)

    print("Segment dataframe head:")
    print(segment_df.head(10))
    print("\nRepetition test summary:")
    print(summary)
    print("\nDiff rows head:")
    print(diff_df.head(10))