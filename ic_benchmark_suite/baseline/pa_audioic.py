from . import Baseline
from pa_audioic.audioic import ICCalcHelper, IntegrationParams
import datasets
from tqdm import tqdm
import soundfile as sf
from io import BytesIO
from librosa import resample
class PAAudioIC(Baseline):
    def __init__(self, device='cuda', integration_params: IntegrationParams = IntegrationParams(), noise_levels = [0.0], monte_carlo_samples = 16):
        super().__init__()
        self.device = device
        self.integration_params = integration_params
        self.ic_calc_helper = ICCalcHelper(device=device)
        self.noise_levels = noise_levels
        self.monte_carlo_samples = monte_carlo_samples
    def ic(self, audio_ds : datasets.Dataset):
        audio_type = 'music'
        results = []
        for ex in tqdm(audio_ds, desc="Computing IC"):
            for instrument, audio in ex['audios'].items():
                if isinstance(audio, bytes):
                    audio, sr = sf.read(BytesIO(audio), dtype="float32", always_2d=False)
                    if sr != 44100:
                        audio = resample(audio, sr, 44100)
                heading_nan_pad, trailing_nan_padding, latents = self.ic_calc_helper.encode_m2l(audio, silence_extractor=audio_type)
                ic = self.ic_calc_helper.ic([heading_nan_pad], [trailing_nan_padding], [latents], noise_levels=self.noise_levels, integration_params=self.integration_params)
                result = {**{f'ic_{nl}': ic[0][i]  for i, nl in enumerate(self.noise_levels) }, 'id': ex['id'], 'instrument': instrument}
                results.append(result)
        entr_ds = datasets.Dataset.from_list(results)
        return entr_ds        
    def entr(self, audio_ds : datasets.Dataset):
        audio_type = 'music'
        results = []
        for ex in tqdm(audio_ds, desc="Computing Entropy"):
            for instrument, mp3_bytes in ex['audios'].items():
                audio, sr = sf.read(BytesIO(mp3_bytes), dtype="float32", always_2d=False)
                heading_nan_pad, trailing_nan_padding, latents = self.ic_calc_helper.encode_m2l(audio, silence_extractor=audio_type)
                entr = self.ic_calc_helper.entr([heading_nan_pad], [trailing_nan_padding], [latents], noise_levels=self.noise_levels, integration_params=self.integration_params, monte_carlo_samples=self.monte_carlo_samples)
                result = {**{f'entr_{nl}': entr[0][i]  for i, nl in enumerate(self.noise_levels) }, 'id': ex['id'], 'instrument': instrument}
                results.append(result)
        entr_ds = datasets.Dataset.from_list(results)
        return entr_ds        
