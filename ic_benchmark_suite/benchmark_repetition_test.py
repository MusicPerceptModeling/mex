from functools import reduce
import math
import re
from typing import Dict, List, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from datasets import Dataset, Features, Sequence, Value, List as FList, concatenate_datasets
from .welford import merge as welford_merge, std as welford_std

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

def _metadata_scheme(annotation_names: List[str]) -> Features:
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
def _model_output_scheme(labels: List[str]) -> Features:
    return Features(
        {
            "id": Value("int64"),
            **{label: Sequence(Value("float64")) for label in labels}
        }
    )

def _extract_segment_df_from_nll_dataset(
    nll_dataset : Dataset,
    segment_names: List[str],
    period_frames: float,
    # midm_column: str = "nll",
    midms: List[str],
    time_offset_frame: int = 0,
    normalize: bool = True,
    num_proc = None,
    exclude_segments = ['Z', 'Silence', 'End'],
    merge_variations = False,
):
    """
    Build a segment-level table equivalent to the logic used in eval.py.

    Expected row fields in nll_dataset:
    - id, midm_column, is_silence
    - optional: noise_level
    - each segment name in segment_names with value dict or list[dict]
      containing start_sec, end_sec, type, optional anidx
    """

    mapped = nll_dataset.map(
        _segment_agg_fn,
        fn_kwargs=dict(
            segment_names=segment_names,
            period_frames=period_frames,
            midm_columns=midms,
            time_offset_frame=time_offset_frame,
            merge_variations=merge_variations
        ),
        keep_in_memory=True,
        num_proc=num_proc,
        remove_columns=nll_dataset.column_names,
    )
    all_results = [item for batch in mapped["all_res"] for item in batch]
    segment_df = pd.DataFrame.from_records(all_results)
    exclude_segments_norm = {str(s).casefold() for s in exclude_segments}
    segment_df = segment_df[
        ~segment_df["type"].astype(str).str.casefold().isin(exclude_segments_norm)
    ]
    if normalize:
        for midm_column in segment_df["midm_column"].unique():
            mask = segment_df["midm_column"] == midm_column
            df = segment_df.loc[mask, ["numel", "mean", "welford"]]
            welford_state = reduce(welford_merge, zip(df["numel"], df["mean"], df["welford"]))
            mean = welford_state[1]
            std = welford_std(welford_state)
            if std == 0:
                segment_df.loc[mask, "mean"] = 0.0
            else:
                segment_df.loc[mask, "mean"] = (segment_df.loc[mask, "mean"] - mean) / std
    if segment_df.empty:
        return segment_df
    segment_df = segment_df.drop(columns=["numel", "welford"])

    return segment_df

def _segment_agg_fn(ex, segment_names, period_frames, midm_columns, time_offset_frame, merge_variations = False):
    outputs = []
    for midm_column in midm_columns:
        if midm_column not in ex:
            raise ValueError(f"Missing required field '{midm_column}' in example")

        nlls = np.asarray(ex[midm_column])
        if "is_silence" in ex:
            is_silence = np.asarray(ex["is_silence"], dtype=bool)
            nlls[is_silence] = np.nan
            
        for seg_name in segment_names:
            segs = ex[seg_name]

            for segment in segs:
                end_secs = np.asarray(segment["end_sec"])
                start_secs = np.asarray(segment["start_sec"])
                types = np.asarray(segment["type"])

                frames = math.ceil(np.nanmax(end_secs) / period_frames)
                pad_len = max(frames - len(nlls), 0)
                if pad_len > 0:
                    pad = np.empty((pad_len,))
                    pad.fill(np.nan)
                    nlls_padded = np.concatenate([nlls, pad], axis=0)
                else:
                    nlls_padded = nlls

                type_counts = {}
                for i, (start, end, type_) in enumerate(zip(start_secs, end_secs, types)):
                    type_ = str(type_)
                    if merge_variations:
                        type_ = type_.strip("'")
                    
                    start_frame = max(0, round(start / period_frames) - time_offset_frame)
                    end_frame = max(0, round(end / period_frames) - time_offset_frame)
                    nll_window = nlls_padded[start_frame:end_frame+1]
                    empty_window = np.array(np.nan) #np.empty((), dtype=nlls_padded.dtype)
                    if len(nll_window) > 0:
                        mask = ~np.isnan(nll_window)
                        numel = np.sum(mask, axis=0)
                        mean_nll = np.mean(nll_window, axis=0, where=mask) if numel > 0 else empty_window                            
                        min_nll = np.min(nll_window, axis=0, where=mask, initial=float('inf')) if numel > 0 else empty_window
                        max_nll = np.max(nll_window, axis=0, where=mask, initial=float('-inf')) if numel > 0 else empty_window
                        # NOTE: for standardization.
                        welford = np.sum((nll_window[mask] - mean_nll) ** 2, axis=0)


                    else:
                        numel = np.array(0)
                        mean_nll = empty_window
                        min_nll = empty_window
                        max_nll = empty_window
                        welford = empty_window

                    nrep = type_counts.setdefault(type_, 0)
                    type_counts[type_] += 1

                    outputs.append( {
                            "midm_column": midm_column,
                            "id": ex["id"],
                            "seg_name": seg_name,
                            "anidx": segment.get("anidx", "-1"),
                            "type": type_,
                            "nrep": nrep,
                            "nseg": i,
                            "mean": mean_nll.item(),
                            "min": min_nll.item(),
                            "max": max_nll.item(),
                            "numel": numel.item(),
                            "welford": welford.item(),
                    })
                    
    return {'all_res': outputs}


def run_repetition_test(
    segment_df,
    midm_column: str | None = None,
    repetition_modes = [['all'], ['first'],],
    ):
    """Vectorized repetition test with optional per-label results.

    When midm_column is None, behaves like the previous single-label function and
    returns: dict[seg_name] -> (summary, residual_df).

    When midm_column is set, returns nested results:
    dict[value] -> dict[seg_name] -> (summary, residual_df).
    """
    df = segment_df.copy()

    required_cols = {"id", "type", "nrep", "mean", "seg_name", "anidx"}
    if midm_column is not None:
        required_cols.add(midm_column)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"segment_df is missing required columns: {sorted(missing)}")

    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")

    sort_cols = ["seg_name", "id", "type", "anidx", "nrep"]
    diff_group_cols = ["seg_name", "id", "type", "anidx"]
    agg_group_cols = ["seg_name"]

    if midm_column is not None:
        sort_cols = [midm_column] + sort_cols
        diff_group_cols = [midm_column] + diff_group_cols
        agg_group_cols = [midm_column] + agg_group_cols

    df = df.sort_values(sort_cols)
    df["diff_val"] = df.groupby(diff_group_cols)["mean"].diff()
    df = df.sort_values([midm_column] + ["seg_name", "id", "anidx", "nseg"])
    
    dfs = []
    for mode in repetition_modes:
        mask = ~df["diff_val"].isna()
        if not 'all' in mode:
            if 'contiguous' in mode:
                same_as_prev = df["type"].eq(df["type"].shift(1))
                mask &= same_as_prev
            if 'first' in mode:
                # df.loc[df.nrep != 1, ['diff_val']] = np.nan
                mask &= df.nrep == 1
        # res = df.dropna(subset=["diff_val"]).copy()
        res = df.loc[mask]

        grouped = res.groupby(agg_group_cols)["diff_val"]
        stats_df = grouped.agg(["mean", "std", "count"]).reset_index()
        stats_df["df"] = stats_df["count"] - 1
        stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["count"])

        with np.errstate(divide="ignore", invalid="ignore"):
            stats_df["t_stat"] = stats_df["mean"] / stats_df["se"]

        # One-sided (less) p-value and confidence bound using t-distribution.
        stats_df["p_value"] = scipy.stats.t.cdf(stats_df["t_stat"], df=stats_df["df"])
        t_crit = scipy.stats.t.ppf(0.95, df=stats_df["df"])
        stats_df["conf_low"] = -np.inf
        stats_df["conf_high"] = stats_df["mean"] + t_crit * stats_df["se"]
        stats_df['mode'] = ",".join(mode)
        dfs.append(stats_df)
    dfs = pd.concat(dfs, ignore_index=True)
    dfs = dfs[ ['midm_column', 'mode', 'seg_name', 'mean', 'p_value']].sort_values(['midm_column', 'mode', 'seg_name'])
    # d = (
    #     dfs.loc[dfs["midm_column"].eq("ENTROPY"), ["mode", "seg_name", "mean"]]
    #     .assign(
    #         mode=lambda x: x["mode"].map({"all": "ALL", "first": "FI"}),
    #         seg_name=lambda x: x["seg_name"].map(
    #             {"functions": "Func.", "uppercase": "Upper.", "lowercase": "Lower."}
    #         ),
    #         row=r"$\mathcal{H}$",
    #     )
    # )

    # # Segment -> Mode (only Δ)
    # wide = (
    #     d.pivot(index="row", columns=["seg_name", "mode"], values="mean")
    #     .reindex(columns=pd.MultiIndex.from_product([["Func.", "Upper.", "Lower."], ["ALL", "FI"]]))
    # )

    # latex = wide.to_latex(
    #     escape=False,
    #     multicolumn=True,
    #     multirow=True,
    #     float_format=lambda x: f"{x:.3f}".replace("0.", "."),  # 2 decimals + decimal comma
    # )

    # print(latex)
    return dfs

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
    return Dataset.from_list(rows, features=_metadata_scheme(["functions", "lowercase", "uppercase"]))


def repetition_decrease(
    model_outputs : Dataset,
    metadata_ds: Dataset,
    periode_frames : float,
    time_offset_frame : int = 0,
    normalize: bool = True,
    exclude_segments = ['Z', 'Silence', 'End', 'no_function'],
    merge_variations = False,
    num_proc = None,
    repetition_modes = [['all'], ['first'],],
):
    labels = [col for col in model_outputs.column_names if col != "id"]
    annotation_names = [col for col in metadata_ds.column_names if col != "id"]
    try:
        scheme = _model_output_scheme(labels)
        model_outputs = model_outputs.cast(scheme)
    except Exception:
        raise ValueError(f"Error occurred while casting model_outputs. Expected Dataset scheme:\n{scheme}.\nActual scheme:\n{model_outputs.features}")
    try:
        scheme = _metadata_scheme(annotation_names)
        metadata_ds = metadata_ds.cast(scheme)
    except Exception:
        raise ValueError(f"Error occurred while casting metadata_ds. Expected Dataset scheme:\n{scheme}.\nActual scheme:\n{metadata_ds.features}")
    # metadata_df = metadata_ds.to_pandas()
    ids_union = set(model_outputs["id"]).intersection(set(metadata_ds["id"]))
    metadata_ds = metadata_ds.sort("id").filter(lambda id: id in ids_union, input_columns=["id"])
    model_outputs = model_outputs.sort("id").filter(lambda id: id in ids_union, input_columns=["id"])
    merged = concatenate_datasets([model_outputs.remove_columns(["id"]), metadata_ds], axis=1)
    if not labels:
        raise ValueError("labels must not be empty")
    available_columns = set(model_outputs.column_names)
    missing_cols = sorted({label for label in labels if label not in available_columns})
    if missing_cols:
        raise ValueError(f"Missing required columns in nll_dataset: {missing_cols}")

    seg_df = _extract_segment_df_from_nll_dataset(
            nll_dataset=merged,
            segment_names=annotation_names,
            period_frames=periode_frames,
            midms=labels,
            time_offset_frame=time_offset_frame,
            normalize=normalize,
            exclude_segments=exclude_segments,
            merge_variations=merge_variations,
            num_proc=num_proc,
        )
        
    return run_repetition_test(seg_df, midm_column="midm_column", repetition_modes=repetition_modes)

if __name__ == "__main__":
    # labels = ["IC"]
    # name = "salami_pa-audioic_ic-0.0-0.3-0.5"  
    # folder = Path(f'/home/mathias/devel/python3/audioic/')
    nll_csv_dir = Path(f'/home/mathias/devel/python3/audioic/salami-entr-mc32')
    csv_paths = sorted(nll_csv_dir.glob('*.csv'), key=_salami_id_extractor)
    nll_dataset = Dataset.from_list(_load_nll_csvs_as_rows(csv_paths))
    # nll_dataset.save_to_disk(f'cache/metrics/salami_audioic_entropy_mc32')

    labels = ["IC_0.0", "IC_0.3", "IC_0.5"]
    n_proc = 8
    # segment_names = ["lowercase", "uppercase", "functions"]
    salami_root = "/opt/datasets/fs/salami"
    cache_dir = "/home/mathias/devel/python3/audio-it/it_benchmark_suite/cache"
    # TODO, if not provided, we could assume something.
    periode_frames = 4096 / 44100
    salami_root = Path(salami_root)
    
    salami_metadata_ds = read_salami_annotations_hf(
        salami_root,
        cache_dir=cache_dir,
        n_proc=n_proc
    )
    # salami_metadata_ds = salami_metadata_ds.cast(metadata_scheme())
    # salami_metadata_df = salami_metadata_ds.to_pandas()
    
    results_by_label = repetition_decrease(
        nll_dataset,
        salami_metadata_ds,
        periode_frames,
        time_offset_frame=0,
    )
    results = results_by_label[labels[0]]

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