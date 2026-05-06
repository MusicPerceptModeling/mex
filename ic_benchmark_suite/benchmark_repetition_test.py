from functools import reduce
import math
from typing import List

import numpy as np
import pandas as pd
import scipy.stats
from datasets import Dataset, concatenate_datasets

from ic_benchmark_suite.benchmark import PandasTestResult, Task
from ic_benchmark_suite.metadata.segment import SegmentMetaData
from .welford import merge as welford_merge, std as welford_std


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



class RepetitionDecreaseTask(Task):
    def __init__(self):
        super().__init__(SegmentMetaData)

    def _run_task(
        self,
        midm : Dataset,
        metadata: Dataset,
        periode_frames : float,
        time_offset_frame : int = 0,
        normalize: bool = True,
        exclude_segments = ['Z', 'Silence', 'End', 'no_function'],
        merge_variations = False,
        num_proc = None,
        repetition_modes = [['all'], ['first'],],
    ) -> PandasTestResult:
        ids_union = set(midm["id"]).intersection(set(metadata["id"]))
        metadata = metadata.sort("id").filter(lambda id: id in ids_union, input_columns=["id"])
        midm = midm.sort("id").filter(lambda id: id in ids_union, input_columns=["id"])
        merged = concatenate_datasets([midm.remove_columns(["id"]), metadata], axis=1)
        # available_columns = set(midm.column_names)
        # missing_cols = sorted({label for label in labels if label not in available_columns})
        # if missing_cols:
        #     raise ValueError(f"Missing required columns in nll_dataset: {missing_cols}")
        midm_cols = self.midm_cols(midm)
        annotation_names = [col for col in metadata.column_names if col != "id"]
        seg_df = _extract_segment_df_from_nll_dataset(
                nll_dataset=merged,
                segment_names=annotation_names,
                period_frames=periode_frames,
                midms=midm_cols,
                time_offset_frame=time_offset_frame,
                normalize=normalize,
                exclude_segments=exclude_segments,
                merge_variations=merge_variations,
                num_proc=num_proc,
            )
            
        df = run_repetition_test(seg_df, midm_column="midm_column", repetition_modes=repetition_modes)
        return PandasTestResult(summary=df)