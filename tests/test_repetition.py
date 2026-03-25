import numpy as np

from ic_benchmark_suite import repetition_decrease
import pytest
from datasets import Dataset
param = pytest.mark.parametrize


# @pytest.mark.skipif(
#     not torch.cuda.is_available() or \
#     torch.cuda.get_device_capability()[0] < 8 or \
#     __import__('importlib').util.find_spec('flash_attn') is None,
#     reason="CUDA compute capability must be >= 8 and flash_attn must be installed"
# )
# @param('exp', (
#     dict(causal=True, same_partition=True, pos_enc='rotary_pos_emb'),
#     dict(causal=False, same_partition=True, pos_enc='rotary_pos_emb'),
#     dict(causal=False, same_partition=False, pos_enc='rotary_pos_emb'),
#     dict(causal=True, same_partition=True, pos_enc='rotary_xpos'))
# )
def test_repetition():
    icc = np.arange(12)
    model_output = Dataset.from_list([
        {
            'id': 0,
            'icc': icc,
        },
        {
            'id': 1,
            'icc': icc,
        },
    ])
    segment1 = 3*np.arange(4+1)
    metadata_ds = Dataset.from_list([
        {
            'id': 0,
            'seg_anno': [
                {
                "anidx": 0,
                "end_sec": segment1[1:],
                "start_sec": segment1[:-1],
                "type": ["a", "b", "a", "b"]}
            ]
        },
        {
            'id': 1,
            'seg_anno': [
                {
                "anidx": 0,
                "end_sec": segment1[1:],
                "start_sec": segment1[:-1],
                "type": ["a", "b", "b", "a"]
                },
                {
                "anidx": 1,
                "end_sec": segment1[1:],
                "start_sec": segment1[:-1],
                "type": ["a", "a", "a", "b"]
                }
            ]
        },
    ])
    results = repetition_decrease(
        model_outputs=model_output,
        metadata_ds=metadata_ds,
        periode_frames=1.0,
        time_offset_frame=1,
    )
    icc_win = [np.mean(icc[s:e+1])  for s, e in [(0, 2), (2, 5), (5, 8), (8, 11)]]
    diffs = [
        icc_win[2]- icc_win[0], icc_win[3]-icc_win[1], 
        icc_win[3]-icc_win[0], icc_win[2]-icc_win[1],
        icc_win[1]-icc_win[0], icc_win[2]-icc_win[1],
    ]
    mean_diff = np.mean(diffs)
    assert results['icc']['seg_anno'][0]['mean'] == mean_diff