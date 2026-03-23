from io import BytesIO
from pathlib import Path

import datasets
from audioic.audioic import ICCalcHelper, PERIODE_MUSIC_2_LATENT
from pa_audioic.audioic import ICCalcHelper as PAICCalcHelper, IntegrationParams
import soundfile as sf
from tqdm import tqdm
from ic_benchmark_suite import idyom_ic_corr_session, idyom_entropy_corr_session
# import numpy as np
# import matplotlib.pyplot as plt

cache_dir = './ic_benchmark_suite/cache'

if __name__ == '__main__':
    # audioic idyom ic consitency test
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
		periode_frames=PERIODE_MUSIC_2_LATENT,
        labels=['ic'],
        combine_inst=True,
        cache_dir=cache_dir
    )
    print(results)
    
    # pa-audioic idyom entropy consitency test
    
    audioic_session_entr = f'{cache_dir}/ic/audioic_session_pa_entropy_mc_16'
    audio_ds = datasets.load_from_disk(f'{cache_dir}/datasets/session_ds')
    noise_levels = [0.0, 0.3, 0.5, 0.6, 0.9]
    if Path(audioic_session_entr).exists():
        entr_ds = datasets.load_from_disk(audioic_session_entr)
    else:
        # TODO: change to get online
        device = 'cuda'
        monte_carlo_samples = 16
        n_runs = 1
        ic_calc_helper = PAICCalcHelper(device=device)
        integration_params_entr = IntegrationParams(solver='euler', solver_kwargs={'n_steps': 100}, n_runs=n_runs)        
        audio_type = 'music'
        results = []
        for ex in tqdm(audio_ds, desc="Computing Entropy"):
            for instrument, mp3_bytes in ex['audios'].items():
                audio, sr = sf.read(BytesIO(mp3_bytes), dtype="float32", always_2d=False)
                heading_nan_pad, trailing_nan_padding, latents = ic_calc_helper.encode_m2l(audio, silence_extractor=audio_type)
                entrs = ic_calc_helper.entr([heading_nan_pad], [trailing_nan_padding], [latents], noise_levels=noise_levels, integration_params=integration_params_entr, monte_carlo_samples=monte_carlo_samples)
                result = {**{f'entr_{nl}': entrs[0][i]  for i, nl in enumerate(noise_levels) }, 'id': ex['id'], 'instrument': instrument}
                results.append(result)
        entr_ds = datasets.Dataset.from_list(results)
        entr_ds.save_to_disk(audioic_session_entr)
    # entr_ds = [{**{f'entr_{nl}': ex['ic'][i]  for i, nl in enumerate(noise_levels) }, 'id': ex['id'], 'instrument': ex['instrument']} for ex in entr_ds]
    labels = [ f'entr_{nl}' for nl in noise_levels]
    results = session_results = idyom_entropy_corr_session(
		ic_dataset=entr_ds,
		periode_frames=PERIODE_MUSIC_2_LATENT,
        labels=labels,
        combine_inst=True,
        cache_dir=cache_dir
    )
    print(results)
    
    
    
    
            