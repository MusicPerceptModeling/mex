from pathlib import Path
from typing import Any, Iterable, List, Mapping

import einops
import numpy as np
import pandas as pd
import scipy.stats
from datasets import Dataset, Value, List as FList, Features

from ic_benchmark_suite.benchmark import Task, PandasTestResult
from ic_benchmark_suite.metadata.symbolic import IDyOMMetaData


class IDyOMConsistencyTask(Task):
    def __init__(self):
        super().__init__(IDyOMMetaData)

    def _run_task(
        self,
        midm : Dataset,
        metadata: Dataset,
        periode_frames : float,
        remove_outliers: bool = False, # TODO: check what we do with 
    ) -> PandasTestResult:
        df = self.idyom_corr(
            val_dataset=midm,
            idyom_ds=metadata,
            periode_frames=periode_frames,
            remove_outliers = remove_outliers
        )
        return PandasTestResult(summary=df)
    def parse_midm_scheme(self, dataset: Dataset):
        assert "id" in dataset.column_names, "Dataset must contain 'id' column."
        midm_cols = self.midm_cols(dataset)
        if not midm_cols:
            raise ValueError("labels must not be empty")
        scheme = self._midm_scheme(midm_cols)
        scheme['instrument'] = Value('string')
        # assert scheme == {feature: dataset.features[feature] for feature in scheme}, f"Dataset features {dataset.features} do not match required features {scheme} for midm."
        dataset.select_columns(list(scheme.keys())).cast(scheme)

    @classmethod
    def idyom_corr(
        cls,
        val_dataset: Dataset | Iterable[Mapping[str, Any]],
        idyom_ds : Dataset,
        periode_frames: float,
        # combine_inst: bool = True,
        remove_outliers: bool = False,
    ):
        # idyom_ds = idyom_ds.cast(Features({'id': Value('int64'), 'idyom_val': FList(Value('float64')), 'onset': FList(Value('float64'))}))
        dfs = []
        for idyom_midm in ['idyom_ic', 'idyom_entropy']:
            if idyom_midm not in idyom_ds.column_names:
                continue
            idyom_df = idyom_ds.select_columns([idyom_midm, 'onset', 'id']).to_pandas().set_index('id')
            labels = [col for col in val_dataset.column_names if col not in ['id', 'instrument']]
            # val_dataset = val_dataset.cast(Features({**{col: FList(Value('float64')) for col in labels}, 'id': Value('int64'), 'instrument': Value('string')}))
            idyom = []
            nlls = []
            ic_dataset_df = pd.DataFrame(val_dataset).sort_values(['id','instrument']).set_index('id')
            idyom_df = idyom_df.sort_index()
            for song_id, row in idyom_df.iterrows():
                idyom_val = row[idyom_midm]
                idyom_time = row['onset']
                if len(idyom_val) != len(idyom_time) or not song_id in ic_dataset_df.index :
                    continue
                val = ic_dataset_df.loc[song_id][labels].values
                min_len = min(len(val_) for val_list in val for val_ in val_list)
                val = np.array([[val_[:min_len]  for val_ in val_list] for val_list in val])

                T = val.shape[-1]
                time = np.arange(T) * periode_frames
                idyom_mask = np.ones_like(idyom_val, dtype=bool)
                if remove_outliers:
                    # remove too extreme values too
                    q = np.quantile(idyom_val[idyom_mask], 0.99)
                    idyom_mask = idyom_mask & (idyom_val < q)
                onset_times = idyom_time[idyom_mask]
                quantized_onset_times = np.searchsorted(time, onset_times)
                vals_from_ours = []
                window_size = 1 # NOTE: window of two nearest frames.
                for j, i in zip(np.where(idyom_mask)[0], quantized_onset_times):
                    window = val[..., max(0, i - window_size) : i + window_size]
                    if np.isnan(window).any():
                        idyom_mask[j] = False
                    else:
                        window = window.mean(-1)
                        vals_from_ours.append(window)
                vals_ours_ret = np.stack(vals_from_ours, -1)
                idyom_ret = idyom_val[idyom_mask]
                idyom.append(idyom_ret)
                nlls.append(vals_ours_ret)
            idyom_all = np.concatenate(idyom, -1)
            nlls_all = np.concatenate(nlls, -1)
            results = []
            for combine_inst in [True, False]:
                if combine_inst:
                    n_inst = nlls_all.shape[0]
                    nlls = einops.rearrange(nlls_all, 'n_inst n_labels t -> 1 n_labels (n_inst t )')
                    idyom = einops.repeat(idyom_all, 't -> (n_inst t)', n_inst=n_inst)
                    instruments = ["combined"]
                else:
                    instruments = ic_dataset_df.instrument.unique()
                    nlls = nlls_all
                    idyom = idyom_all
                for instrument, nll_inst in zip(instruments, nlls):
                    for label, nll_label in zip(labels, nll_inst):
                        r_val, p_val = scipy.stats.spearmanr(nll_label, idyom)
                        results.append({'instrument': instrument, 'label': label, 'r_val': r_val, 'p_val': p_val})
            def _fmt(x: float, ndigits: int = 3) -> str:
                if pd.isna(x):
                    return ".xxx"
                s = f"{x:.{ndigits}f}"
                return s.replace("-0.", "-.").replace("0.", ".")

            def make_latex_row(
                df: pd.DataFrame,
                label: str = "ic",          # e.g. "ic" or "entropy"
                row_name: str = "GIVT",     # text in 2nd column
                value_col: str = "r_val",
            ) -> str:
                inst_order = [
                    "Acoustic Grand Piano",
                    "Acoustic Guitar (nylon)",
                    "Pad 1 (new age)",
                    "Synth Voice",
                    "Trumpet",
                    "combined",
                ]

                # Keep only the label you want
                d = df[df["label"] == label].copy()

                # Model-agnostic: if multiple rows per instrument (e.g. multiple models), average them
                d = d.groupby("instrument", as_index=False)[value_col].mean()

                values = d.set_index("instrument")[value_col]
                cells = [_fmt(values.get(inst, np.nan)) for inst in inst_order]

                return f"& {row_name} & " + " & ".join(cells) + r" \\"


            df = pd.DataFrame.from_records(results).sort_values(['label', 'instrument'])
            df['idyom_midm'] = idyom_midm
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df