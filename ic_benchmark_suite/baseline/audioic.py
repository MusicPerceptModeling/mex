from . import Baseline
from audioic.audioic import ICCalcHelper
import datasets
from tqdm import tqdm
import soundfile as sf
from io import BytesIO
from librosa import resample

class AudioIC(Baseline):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.ic_calc_helper = ICCalcHelper(device=device)
    def ic(self, audio_ds : datasets.Dataset):
        audio_type = 'music'
        results = []
        for ex in tqdm(audio_ds, desc="Computing IC"):
            for instrument, audio in ex['audios'].items():
                if isinstance(audio, bytes):
                    audio, sr = sf.read(BytesIO(audio), dtype="float32", always_2d=False)
                    if sr != 44100:
                        audio = resample(audio, sr, 44100)
                heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded = self.ic_calc_helper.encode_m2l(audio, silence_extractor=audio_type)
                ic = self.ic_calc_helper.ic(heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded)
                ic = ic[0]
                results.append({'id': ex['id'], 'instrument': instrument, 'ic': ic})
        ic_ds = datasets.Dataset.from_list(results)
        return ic_ds
    @classmethod
    def periode_frames(cls):
        return 4096/44100