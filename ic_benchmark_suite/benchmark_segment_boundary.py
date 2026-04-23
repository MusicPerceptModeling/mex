import json

import pandas as pd

from ic_benchmark_suite.dataset_provider import DatasetProvider
from ic_benchmark_suite import peak_picking
from datasets import Dataset, concatenate_datasets
import numpy as np
from scipy import ndimage
import mir_eval
from typing import List
def transform_for_peak(std, nll):
    nll[np.isnan(nll)] = np.nanmin(nll)
    nll = ndimage.gaussian_filter1d(nll, std)

    nll = np.diff(nll, prepend=nll.min())
    # nll = np.abs(nll)
    return nll

def calc_peaks(ex, hierarchy_levels, periode_frames, window_boundary_matching, time_offset_frame=0, peak_picking_kwargs :List[dict] = [dict(std=5.0)], repetition_modes=['all', 'only_first_occurence', 'change']):
    hierarchy_levels = {level: ex.pop(level) for level in hierarchy_levels}
    nl, nll = zip(*ex.items())
    min_len = min(len(nll_) for nll_ in nll)
    nll = np.stack([nll_[:min_len] for nll_ in nll], axis=0)
    n_cols, len_seqs = nll.shape
    res_all = {}
    for peak_picking_kwargs_ in peak_picking_kwargs:
        nll = transform_for_peak(peak_picking_kwargs_['std'], nll)
        time = (np.arange(len_seqs) + time_offset_frame) * periode_frames
        peaks = []
        for nll_ in nll:
            peaks_idx = peak_picking.peak_picking_roeder(nll_, direction=None, abs_thresh=None, 
                                        rel_thresh=None, descent_thresh=None, 
                                        tmin=None, tmax=None)
            peaks.append(time[peaks_idx.astype(int)])
        for col, peak in zip(nl, peaks):
            res = res_all.setdefault(col, {})
            for level_name, level in hierarchy_levels.items():
                for anotation in level:
                    for repetition_mode in repetition_modes:
                        secs_start = np.array(anotation['start_sec'])
                        if repetition_mode == 'only_first_occurence':
                            types = np.array(anotation['type'])
                            labs = np.unique(types)
                            peak_idx = np.argmax(types[:, None] == labs[None], axis=0)
                            secs_start = secs_start[peak_idx]
                        elif repetition_mode == 'change':
                            types = np.array(anotation['type'])
                            mask = types[1:] != types[:-1]
                            secs_start = np.append(secs_start[0], secs_start[1:][mask])
                        elif repetition_mode != 'all':
                            raise NotImplementedError("Only 'all', 'only_first_occurence' and 'change' are implemented for repetition_mode.")
                        matches = mir_eval.util.match_events(secs_start, peak, window=window_boundary_matching)
                        res.setdefault('peak_picking_kwargs', []).append(json.dumps(peak_picking_kwargs_))
                        res.setdefault('repetition_mode', []).append(repetition_mode)
                        res.setdefault('hierarchy_level', []).append(level_name)
                        res.setdefault('anidx', []).append(anotation['anidx'])
                        res.setdefault('tp',[]).append(len(matches))
                        res.setdefault('nrefs', []).append(len(secs_start))
                        res.setdefault('npreds', []).append(len(peak))
    return res_all
def segment_boundary_alignment(
    model_outputs : Dataset,
    metadata_ds: Dataset,
    periode_frames : float,
    peak_picking_kwargs :List[dict] = [dict(std=5.0)],
    time_offset_frame : int = 0,
    window_boundary_matching: float = 1.0,
    repetition_modes: list = ['all', 'only_first_occurence', 'change'],
    hierarchy_levels = ['lowercase', 'uppercase', 'functions'],
    num_proc = None,
):
    metadata_ds = metadata_ds.select_columns(['id', *hierarchy_levels])
    model_outputs = model_outputs.sort('id')
    metadata_ds = metadata_ds.sort('id')
    common_ids = set(model_outputs['id']).intersection(set(metadata_ds['id']))
    model_outputs = model_outputs.filter(lambda id: id in common_ids, input_columns=['id'])
    metadata_ds = metadata_ds.filter(lambda id: id in common_ids, input_columns=['id'])
    concat = concatenate_datasets([model_outputs.remove_columns(['id']), metadata_ds], axis=1)
    
    
    rng = np.random.default_rng(42)
    shuffled_ids = rng.permutation(sorted(common_ids))
    split_idx = int(len(shuffled_ids) * 0.8)
    train_ids = set(shuffled_ids[:split_idx])
    test_ids = set(shuffled_ids[split_idx:])
    train = concat.filter(lambda id: id in train_ids, input_columns=['id'])
    test = concat.filter(lambda id: id in test_ids, input_columns=['id'])
    train_res = train.remove_columns(['id']).map(
        calc_peaks,
        fn_kwargs=dict(
            hierarchy_levels=hierarchy_levels,
            peak_picking_kwargs=peak_picking_kwargs,
            periode_frames=periode_frames,
            window_boundary_matching=window_boundary_matching,
            time_offset_frame=time_offset_frame,
            repetition_modes=repetition_modes
        ),
        keep_in_memory=True,
        num_proc=num_proc
    )
    
    train_dfs = extract_performance(train_res)
    best_peak_picking = (
        train_dfs.assign(f_measure=pd.to_numeric(train_dfs['f_measure'], errors='coerce'))
        .sort_values('f_measure', ascending=False)
        .drop_duplicates(['hierarchy_level', 'repetition_mode', 'midm'], keep='first')
        .loc[:, ['hierarchy_level', 'repetition_mode', 'midm', 'peak_picking_kwargs', 'f_measure']]
        .sort_values(['midm', 'hierarchy_level', 'repetition_mode'])
        .reset_index(drop=True)
    )
    test_res = test.remove_columns(['id']).map(
        calc_peaks,
        fn_kwargs=dict(
            hierarchy_levels=hierarchy_levels,
            peak_picking_kwargs=peak_picking_kwargs,
            periode_frames=periode_frames,
            window_boundary_matching=window_boundary_matching,
            time_offset_frame=time_offset_frame,
            repetition_modes=repetition_modes
        ),
        keep_in_memory=True,
        num_proc=num_proc
    )
    test_dfs = extract_performance(test_res)
    selected_test = (
        test_dfs.merge(
            best_peak_picking[['hierarchy_level', 'repetition_mode', 'midm', 'peak_picking_kwargs']],
            on=['hierarchy_level', 'repetition_mode', 'midm', 'peak_picking_kwargs'],
            how='inner'
        )
        .loc[:, ['hierarchy_level', 'repetition_mode', 'midm', 'peak_picking_kwargs', 'precision', 'recall', 'f_measure']]
        .sort_values(['midm', 'hierarchy_level', 'repetition_mode'])
        .reset_index(drop=True)
    )

    return selected_test

def extract_performance(res_all_):
    dfs = []
    
    for midm in res_all_.column_names:
        res_all = res_all_[midm]
        df = pd.DataFrame(res_all)
        df = df.explode(list(df.columns))
        def compute_metrics(group):
            tps = group['tp'].sum()
            n_preds = group['npreds'].sum()
            nrefs = group['nrefs'].sum()
            precision = tps/n_preds if n_preds > 0 else 0.0
            recall =  tps/nrefs if nrefs > 0 else 0.0
            f_measure = mir_eval.util.f_measure(precision, recall, beta=1)
            return pd.Series({'precision': precision, 'recall': recall, 'f_measure': f_measure})
        dfs.append(df.groupby(['hierarchy_level', 'repetition_mode', 'peak_picking_kwargs']).apply(compute_metrics).reset_index())
    dfs = pd.concat(dfs, keys=res_all_.column_names)
    dfs = dfs.reset_index(0, names='midm')
    return dfs

# if __name__ == '__main__':
#     dataset_provider = DatasetProvider()
#     from pathlib import Path
#     root = Path(__file__).parent.parent
#     ic_ds = dataset_provider.get_ds(f'{root}/cache/ic/audioic_salami_ic')
#     # ic_ds = dataset_provider.get_ds(f'{root}/cache/ic/pa_audioic_salami_ic')
#     ic_ds = ic_ds.remove_columns('Time')
#     salami_ds = dataset_provider.get_ds(f'{root}/cache/datasets/salami_metadata_ds')
#     sweep_result = segment_boundary_alignment(ic_ds, salami_ds, 4096/44100, 1, segment_inclusion='change', labels='lowercase')
#     # no abs
#     pd.DataFrame(sweep_result, columns=['type', 'noise_level', 'std', 'precision', 'recall', 'f_measure'])