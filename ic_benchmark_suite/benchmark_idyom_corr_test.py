import math
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping

import einops
import numpy as np
import pandas as pd
import scipy.stats
import soundfile as sf
from datasets import Dataset
from datasets import load_from_disk

# def _load_session_midi_files(
# 	midi_path: Path = Path("/home/mathias/devel/python3/DSTM/data/session/"),
# 	session_split_path: Path = Path("session_split.json"),
# 	n_midis: int | None = None,
# ) -> list[Path]:
# 	"""Load session MIDI file paths from the test split, matching notebook setup."""
# 	split = sorted(json.load(open(session_split_path))["test"])
# 	midi_files = [Path(midi_path, f"sessiontune{id_}.mid") for id_ in split]
# 	midi_files = [p for p in midi_files if p.exists()]
# 	if n_midis is not None:
# 		midi_files = midi_files[:n_midis]
# 	return midi_files


program_dict = {
	"Acoustic Grand Piano": 0, "Bright Acoustic Piano": 1, "Electric Grand Piano": 2, "Honky-tonk Piano": 3,
	"Electric Piano 1": 4, "Electric Piano 2": 5, "Harpsichord": 6, "Clavinet": 7,
	"Celesta": 8, "Glockenspiel": 9, "Music Box": 10, "Vibraphone": 11,
	"Marimba": 12, "Xylophone": 13, "Tubular Bells": 14, "Dulcimer": 15,
	"Drawbar Organ": 16, "Percussive Organ": 17, "Rock Organ": 18, "Church Organ": 19,
	"Reed Organ": 20, "Accordion": 21, "Harmonica": 22, "Tango Accordion": 23,
	"Acoustic Guitar (nylon)": 24, "Acoustic Guitar (steel)": 25, "Electric Guitar (jazz)": 26, "Electric Guitar (clean)": 27,
	"Electric Guitar (muted)": 28, "Overdriven Guitar": 29, "Distortion Guitar": 30, "Guitar harmonics": 31,
	"Acoustic Bass": 32, "Electric Bass (finger)": 33, "Electric Bass (pick)": 34, "Fretless Bass": 35,
	"Slap Bass 1": 36, "Slap Bass 2": 37, "Synth Bass 1": 38, "Synth Bass 2": 39,
	"Violin": 40, "Viola": 41, "Cello": 42, "Contrabass": 43,
	"Tremolo Strings": 44, "Pizzicato Strings": 45, "Orchestral Harp": 46, "Timpani": 47,
	"String Ensemble 1": 48, "String Ensemble 2": 49, "SynthStrings 1": 50, "SynthStrings 2": 51,
	"Choir Aahs": 52, "Voice Oohs": 53, "Synth Voice": 54, "Orchestra Hit": 55,
	"Trumpet": 56, "Trombone": 57, "Tuba": 58, "Muted Trumpet": 59,
	"French Horn": 60, "Brass Section": 61, "SynthBrass 1": 62, "SynthBrass 2": 63,
	"Soprano Sax": 64, "Alto Sax": 65, "Tenor Sax": 66, "Baritone Sax": 67,
	"Oboe": 68, "English Horn": 69, "Bassoon": 70, "Clarinet": 71,
	"Piccolo": 72, "Flute": 73, "Recorder": 74, "Pan Flute": 75,
	"Blown Bottle": 76, "Shakuhachi": 77, "Whistle": 78, "Ocarina": 79,
	"Lead 1 (square)": 80, "Lead 2 (sawtooth)": 81, "Lead 3 (calliope)": 82, "Lead 4 (chiff)": 83,
	"Lead 5 (charang)": 84, "Lead 6 (voice)": 85, "Lead 7 (fifths)": 86, "Lead 8 (bass + lead)": 87,
	"Pad 1 (new age)": 88, "Pad 2 (warm)": 89, "Pad 3 (polysynth)": 90, "Pad 4 (choir)": 91,
	"Pad 5 (bowed)": 92, "Pad 6 (metallic)": 93, "Pad 7 (halo)": 94, "Pad 8 (sweep)": 95,
	"FX 1 (rain)": 96, "FX 2 (soundtrack)": 97, "FX 3 (crystal)": 98, "FX 4 (atmosphere)": 99,
	"FX 5 (brightness)": 100, "FX 6 (goblins)": 101, "FX 7 (echoes)": 102, "FX 8 (sci-fi)": 103,
	"Sitar": 104, "Banjo": 105, "Shamisen": 106, "Koto": 107,
	"Kalimba": 108, "Bag pipe": 109, "Fiddle": 110, "Shanai": 111,
	"Tinkle Bell": 112, "Agogo": 113, "Steel Drums": 114, "Woodblock": 115,
	"Taiko Drum": 116, "Melodic Tom": 117, "Synth Drum": 118, "Reverse Cymbal": 119,
	"Guitar Fret Noise": 120, "Breath Noise": 121, "Seashore": 122, "Bird Tweet": 123,
	"Telephone Ring": 124, "Helicopter": 125, "Applause": 126, "Gunshot": 127,
}

def synth_midi(
    midi_data,
    synth_inst
):
    audios = []
    midi_inst_id = program_dict[synth_inst]
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        instrument.program = midi_inst_id
        audios.append(midi_data.fluidsynth(fs=44100))
    audio = np.mean(audios, 0)
    return audio

def _iter_midis(midi_files: list[str], instruments, id_path_extractor):
	"""Yield note onset rows for session MIDI files."""
	import pretty_midi
	instruments = instruments['instruments'] if isinstance(instruments, dict) else instruments
	for midi_path in midi_files:
		p = Path(midi_path)
		id = id_path_extractor(midi_path)
		midi = pretty_midi.PrettyMIDI(str(p))
		if not midi.instruments:
			continue
		audios = {}
		for synth_inst in instruments:
			audio = synth_midi(midi, synth_inst)
			buf = BytesIO()
			sf.write(buf, audio, 44100, format="MP3")
			audios[synth_inst] = buf.getvalue()

		yield {
				"id": id,
				"onset": [note.start for instr in midi.instruments for note in instr.notes],
				"audios": audios,
		}

def load_idyom_from_dat_dir(
	dat_dir: Path,
	midi_dir: list[Path],
	id_path_extractor: Callable[[str], int|str],
	cache_dir: Path | None = None,
	num_proc: int | None = None,
	instruments: list[str]  = ['Pad 1 (new age)', 'Synth Voice','Acoustic Guitar (nylon)', 'Acoustic Grand Piano','Trumpet'],
	n_midis: int | None = None,
) -> pd.DataFrame:
	"""Load session synthetic IDyOM using the notebook's merged_df workflow.

	This follows notebooks/ismir25_iccorr.ipynb:
	1) concat all *.dat into df_idyom_session
	2) derive id from melody.name
	3) build session_onsets from SynthDataset midi files
	4) merged_ = merge(df_idyom_session[['id','note.id','cpitch.information.content']],
	                 session_onsets, on=['id','note.id'])
	5) merged_df = merged_.set_index(['id','note.id'])

	Returns per-session rows with: id, idyom_ic, idyom_onset.
	"""
 
 
	dat_paths = [str(p) for p in sorted(dat_dir.glob("*.dat"))]

	def _iter_dat_rows(paths):
		for p in paths:
			yield from pd.read_csv(p)[['melody.name', 'note.id', 'information.content', 'entropy']].to_dict("records")

	ds_idyom_session = Dataset.from_generator(
    	_iter_dat_rows,
    	gen_kwargs={"paths": dat_paths},
    	cache_dir=cache_dir,
    	num_proc=num_proc,

	)
	ds_idyom_session = ds_idyom_session.map(lambda x: {"id": id_path_extractor(x)}, num_proc=num_proc, input_columns=["melody.name"], remove_columns=["melody.name"])
	df_idyom_session = (
		ds_idyom_session.to_pandas().sort_values(["id", "note.id"])        # optional: control order inside array
		.groupby("id", as_index=False)
		.agg(
      		idyom_ic=("information.content", list),
			idyom_entropy=("entropy", list)
		)
	)
	# df_idyom_session = ds_idyom_session.to_pandas().sort_values(["id", "note.id"]) 
	ids = df_idyom_session.id.unique()
	ids = ids[:n_midis]
	df_idyom_session = df_idyom_session[df_idyom_session.id.isin(ids)]
	midi_files = np.array(list(Path(midi_dir).glob("*.mid")))
	midi_ids = np.array([id_path_extractor(str(p)) for p in midi_files])
	mapper = np.argmax(midi_ids[None, :] == ids[:, None], axis=1)[:n_midis]
	assert (midi_ids[mapper] == ids).all()
	midi_files = midi_files[mapper]
	gen_kwargs = dict(
     	midi_files=[str(p) for p in midi_files],
		instruments={'instruments': instruments},
		id_path_extractor=id_path_extractor,
	)
	session_onsets_ds = Dataset.from_generator(_iter_midis, gen_kwargs=gen_kwargs, cache_dir=cache_dir, num_proc=num_proc)
	concat = session_onsets_ds.add_column("idyom_ic", df_idyom_session["idyom_ic"]).add_column("idyom_entropy", df_idyom_session["idyom_entropy"])
	return concat

def idyom_ic_corr_session(ic_dataset: Dataset | Iterable[Mapping[str, Any]], periode_frames: float, labels: list[str], combine_inst: bool = True, remove_outliers: bool = False, cache_dir: str = '.'):
    audio_ds = load_from_disk(f'{cache_dir}/datasets/session_ds')
    idyom_df = audio_ds.remove_columns(['audios']).to_pandas().set_index('id')
    idyom_df = idyom_df.drop(columns=['idyom_entropy'])
    results =  idyom_corr(
		ic_dataset=ic_dataset,
		idyom_df=idyom_df,
		periode_frames=periode_frames,
        labels=labels,
        combine_inst=combine_inst,
        remove_outliers=remove_outliers,
    )
    return results
def idyom_corr(
	ic_dataset: Dataset | Iterable[Mapping[str, Any]],
 	idyom_df : pd.DataFrame,
	periode_frames: float,
	labels: List[str] = ["ic"],
	combine_inst: bool = True,
    remove_outliers: bool = False,
):
	idyom_all = []
	nlls_all = []
	ic_dataset_df = pd.DataFrame(ic_dataset).sort_values(['id','instrument']).set_index('id')
	idyom_df = idyom_df.sort_index()
	for song_id, (idyom_time, idyom_ic) in idyom_df.iterrows():
		if len(idyom_ic) != len(idyom_time):
			continue
		ic = ic_dataset_df.loc[song_id][labels].values
		min_len = min(len(ic_) for ic_list in ic for ic_ in ic_list)
		ic = np.array([[ic_[:min_len]  for ic_ in ic_list] for ic_list in ic])

		T = ic.shape[-1]
		time = np.arange(T) * periode_frames
		idyom_ic_mask = np.ones_like(idyom_ic, dtype=bool)
		if remove_outliers:
			# remove too extreme values too
			q = np.quantile(idyom_ic[idyom_ic_mask], 0.99)
			idyom_ic_mask = idyom_ic_mask & (idyom_ic < q)
		onset_times = idyom_time[idyom_ic_mask]
		quantized_onset_times = np.searchsorted(time, onset_times)
		ics_from_ours = []
		window_size = 1
		for j, i in zip(np.where(idyom_ic_mask)[0], quantized_onset_times):
			window = ic[..., max(0, i - window_size) : i + window_size]
			if np.isnan(window).any():
				idyom_ic_mask[j] = False
			else:
				window = window.mean(-1)
				ics_from_ours.append(window)
		ics_ours_ret = np.stack(ics_from_ours, -1)
		idyom_ret = idyom_ic[idyom_ic_mask]
		idyom_all.append(idyom_ret)
		nlls_all.append(ics_ours_ret)
	idyom_all = np.concatenate(idyom_all, -1)
	nlls_all = np.concatenate(nlls_all, -1)
	if combine_inst:
		n_inst = nlls_all.shape[0]
		nlls_all = einops.rearrange(nlls_all, 'n_inst n_labels t -> 1 n_labels (n_inst t )')
		idyom_all = einops.repeat(idyom_all, 't -> (n_inst t)', n_inst=n_inst)
		instruments = "combined"
	else:
		instruments = ic_dataset_df.instrument.unique()
	results = []
	for instrument, nll_inst in zip(instruments, nlls_all):
		for label, nll_label in zip(labels, nll_inst):
			r_val, p_val = scipy.stats.spearmanr(nll_label, idyom_all)
			results.append({'instrument': instrument, 'label': label, 'r_val': r_val, 'p_val': p_val})
	return results


def run_idyom_corr_test(
	ic_dataset: Dataset | Iterable[Mapping[str, Any]],
	idyom_df: pd.DataFrame,
	label: str = "nll",
	sr_idyom: float = 64.0,
	period_music_2_latent: float = 0.093,
	idyom_positive_only: bool = True,
):
	"""Single-label wrapper around run_idyom_corr_test_for_labels."""
	return idyom_corr(
		ic_dataset=ic_dataset,
		idyom_df=idyom_df,
		labels=[label],
		sr_idyom=sr_idyom,
		period_music_2_latent=period_music_2_latent,
		idyom_positive_only=idyom_positive_only,
	)


if __name__ == "__main__":
	session_dat_dir = "/home/mathias/devel/python3/audio-it/idyom/session_test_out_full_pitch"
	midi_dir = Path("/home/mathias/devel/python3/DSTM/data/session/")
	cache_dir = "/home/mathias/devel/python3/audio-it/ic_benchmark_suite/cache"
	session_idyom_ds = load_idyom_from_dat_dir(
     	Path(session_dat_dir),
		midi_dir=midi_dir,
		id_path_extractor = lambda x: int(Path(x).stem[len("sessiontune"):]),
		cache_dir=cache_dir,
		n_midis=200,
		num_proc=60,
	)
	session_idyom_ds.save_to_disk(Path(cache_dir, "datasets/session_ds"))
