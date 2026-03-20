from io import BytesIO
from pathlib import Path

import datasets
from audioic.audioic import ICCalcHelper, PERIODE_MUSIC_2_LATENT
import soundfile as sf
from tqdm import tqdm
from ic_benchmark_suite import idyom_ic_corr_session
# import numpy as np
# import matplotlib.pyplot as plt

cache_dir = './ic_benchmark_suite/cache'

if __name__ == '__main__':
    # audioic idyom consitency test
    audioic_session_ic = f'{cache_dir}/ic/audioic_session_ic'
    audio_ds = datasets.load_from_disk(f'{cache_dir}/datasets/session_ds')
    if Path(audioic_session_ic).exists():
        ic_ds = datasets.load_from_disk(audioic_session_ic)
    else:
        # TODO: change to get online
        device = 'cuda'
        ic_calc_helper = ICCalcHelper(device=device)
        audio_type = 'music'
        results = []
        for ex in tqdm(audio_ds, desc="Computing IC"):
            for instrument, mp3_bytes in ex['audios'].items():
                audio, sr = sf.read(BytesIO(mp3_bytes), dtype="float32", always_2d=False)
                heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded = ic_calc_helper.encode_m2l(audio, silence_extractor=audio_type)
                ic = ic_calc_helper.ic(heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded)
                ic = ic[0]
                results.append({'id': ex['id'], 'instrument': instrument, 'ic': ic})
        ic_ds = datasets.Dataset.from_list(results).save_to_disk(audioic_session_ic)
    results = session_results = idyom_ic_corr_session(
		ic_dataset=ic_ds,
		periode_frames=4096/44100,
        labels=['ic'],
        combine_inst=False,
        cache_dir=cache_dir
    )
    print(results)
    
    
            