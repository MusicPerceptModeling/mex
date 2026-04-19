from ic_benchmark_suite.dataset_provider import DatasetProvider
from ic_benchmark_suite import peak_picking
from datasets import Dataset, concatenate_datasets
import numpy as np
from scipy import ndimage
import mir_eval

def transform_for_peak(std, nll):
    nll[np.isnan(nll)] = np.nanmin(nll)
    nll = ndimage.gaussian_filter1d(nll, std)

    nll = np.diff(nll, prepend=nll.min())
    # nll = np.abs(nll)
    return nll

def calc_peaks(ex, labels, periode_frames, time_offset_frame=0, std=4.0, segment_inclusion='all'):
    annotations = ex.pop(labels)
    nl, nll = zip(*ex.items())
    nll = np.stack(nll, axis=0)
    n_cols, len_seqs = nll.shape
    # nll = np.array(ex['nll'])[:, noise_level]
    nll = transform_for_peak(std, nll)
    time = (np.arange(len_seqs) + time_offset_frame) * periode_frames
        # peaks_idx, _ , _ = peak_picking.peak_picking_MSAF(nll_nan_free, sigma=32.0, median_len=32)
        # peaks_idx, smoothed, local_threshold = peak_picking.peak_picking_MSAF(nll_nan_free, sigma=6.0, offset_rel=-2, median_len=40)
        # peaks_idx, smoothed, local_threshold = peak_picking.peak_picking_MSAF(nll_nan_free, sigma = 4.0, offset_rel=0, median_len=32)
    # peaks = time[peaks_idx]
    # peaks = peak_picking.peak_picking_boeck(nll, fps=1/PERIODE_MUSIC_2_LATENT)
    # NOTE: From foote
    peaks = []
    for nll_ in nll:
        peaks_idx = peak_picking.peak_picking_roeder(nll_, direction=None, abs_thresh=None, 
                                      rel_thresh=None, descent_thresh=None, 
                                      tmin=None, tmax=None)
        peaks.append(time[peaks_idx.astype(int)])
        # peaks_idx = [peak_picking.peak_picking_roeder(nll_, direction=None, abs_thresh=None, 
    #                                   rel_thresh=None, descent_thresh=None, 
    #                                   tmin=None, tmax=None) for nll_ in nll]
    
    # peaks = time[peaks_idx.astype(int)]
    # reference_boundaries = mir_eval.util.intervals_to_boundaries(secs_start)
    # estimated_boundaries = mir_eval.util.intervals_to_boundaries(estimated_intervals)
    
   

    # labels_unique, labels_unique_idx = np.unique(types, return_index=True)
    # useful_labels_mask = ~(labels_unique == np.array([['End'], ['Silence']])).any(axis=0)
    # first_occurences = secs_start[labels_unique_idx[useful_labels_mask]]
    # first_occurences.sort()
    # secs_start = first_occurences
    res_all = {}
    for col, peak in zip(nl, peaks):
        res = res_all.setdefault(col, {})
        for ano in annotations:
            secs_start = np.array(ano['start_sec'])
            if segment_inclusion == 'only_first_occurence':
                types = np.array(ano['type'])
                labs = np.unique(types)
                peak_idx = np.argmax(types[:, None] == labs[None], axis=0)
                secs_start = secs_start[peak_idx]
            elif segment_inclusion == 'change':
                types = np.array(ano['type'])
                mask = types[1:] != types[:-1]
                secs_start = np.append(secs_start[0], secs_start[1:][mask])
            elif segment_inclusion != 'all':
                raise NotImplementedError("Only 'all', 'only_first_occurence' and 'change' are implemented for segment_inclusion.")
            matches = mir_eval.util.match_events(secs_start, peak, window=0.5)
            # TODO: probably a lot of this is not needed.
            res.setdefault('anidx', []).append(ano['anidx'])
            res.setdefault('secs_start',[]).append(secs_start)
            res.setdefault('peaks', []).append(peak)
            res.setdefault('proced_nll', []).append(nll)
            res.setdefault('tp',[]).append(len(matches))
            res.setdefault('nrefs', []).append(len(secs_start))
            res.setdefault('npreds', []).append(len(peak))
            # res.setdefault('part', []).append(types)
    return res_all
def segment_boundary_alignment(
    model_outputs : Dataset,
    metadata_ds: Dataset,
    periode_frames : float,
    time_offset_frame : int = 0,
    repetition_mode: str = 'all',
    hiearchy_levels = 'lowercase',
    exclude_segments = ['Z', 'Silence', 'End']
):
    num_proc = None
    metadata_ds = metadata_ds.select_columns(['id', hiearchy_levels])
    model_outputs = model_outputs.sort('id')
    metadata_ds = metadata_ds.sort('id')
    common_ids = set(model_outputs['id']).intersection(set(metadata_ds['id']))
    model_outputs = model_outputs.filter(lambda id: id in common_ids, input_columns=['id'])
    metadata_ds = metadata_ds.filter(lambda id: id in common_ids, input_columns=['id'])
    concat = concatenate_datasets([model_outputs.remove_columns(['id']), metadata_ds], axis=1)
    
    
    # concat = concat.remove_columns([feat for feat in concat.column_names if feat not in ['nll', labels, 'noise_level']])

    std = 5.0
    # for noise_level_index, nl in enumerate(concat[0]['noise_level']):
    res_all_ = concat.remove_columns(['id']).map(calc_peaks, fn_kwargs=dict(labels=hiearchy_levels, std=std, periode_frames=periode_frames, time_offset_frame=time_offset_frame, segment_inclusion=repetition_mode), keep_in_memory=True, num_proc=num_proc)
    # parts = np.array(res_all[type_]['part'])
    sweep_result = []
    for nl in res_all_.column_names:
        res_all = res_all_[nl] 
        # tps = np.array(res_all['tp'])
        # n_preds = np.array(res_all['npreds'])
        # nrefs = np.array(res_all['nrefs'])
        # precision = np.sum(np.concatenate(tps))/np.sum(np.concatenate(n_preds))
        # recall =  np.sum(np.concatenate(tps))/np.sum(np.concatenate(nrefs))
        
        
        tps = sum(sum(r) for r in res_all['tp'])
        n_preds = sum(sum(r) for r in res_all['npreds'])
        nrefs = sum(sum(r) for r in res_all['nrefs'])
        precision = tps/n_preds
        recall =  tps/nrefs
        f_measure = mir_eval.util.f_measure(precision, recall, beta=1)
        # print(Path(config_file).stem, part, std, precision, recall, f_measure)
        sweep_result.append((nl, std, precision, recall, f_measure))
    return sweep_result
    # # fig, ax = plt.subplots(figsize=(16, 5))
    # # plot_ex(types, secs_start,  ax = ax)
    # # ax.plot(nll)
    # # # ax.plot(smoothed, 'g')
    # # # ax.plot(local_threshold, 'cyan')
    # # ax.set_ylim(-20, 20)
    # # ax.vlines(peaks / PERIODE_MUSIC_2_LATENT, -15, 15, color='y')



    # e = 1e-9
    # ref = np.stack([secs_start, secs_start+e], axis=1)
    # ref[:,1] += e
    # preds = np.stack([peaks, peaks+e], axis=1)
    # print(mir_eval.segment.detection(ref, preds, window=.5))

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