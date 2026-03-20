import math
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import einops
import numpy as np
import pandas as pd
import scipy.stats
import soundfile as sf
from datasets import Dataset
from datasets import load_from_disk



def load_idyom_pitch_ic_from_npy(idyom_dir: Path) -> pd.DataFrame:
	"""Load IDyOM pitch IC sequences from *.npy files.

	Expected filename pattern: "{id}_song.npy".
	Returns a DataFrame with columns: id, idyom_ic.
	"""
	rows = []
	for p in sorted(idyom_dir.glob("*.npy")):
		match = re.match(r"^(?P<id>\d+)_song$", p.stem)
		if match is None:
			continue
		rows.append(
			{
				"id": int(match.group("id")),
				"idyom_ic": np.asarray(np.load(p), dtype=float),
			}
		)

	if not rows:
		raise ValueError(f"No IDyOM .npy files found in {idyom_dir}")

	return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


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

def _iter_session_note_onsets(midi_files: list[str], instruments):
	"""Yield note onset rows for session MIDI files."""
	import pretty_midi
	instruments = instruments['instruments'] if isinstance(instruments, dict) else instruments
	for midi_path in midi_files:
		p = Path(midi_path)
		match = re.match(r"^sessiontune(?P<id>\d+)$", p.stem)
		if match is None:
			continue
		session_id = int(match.group("id"))
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
				"session_id": session_id,
				"onset": [note.start for instr in midi.instruments for note in instr.notes],
				"audios": audios,
		}


def load_idyom_session_ic_from_csv(session_csv_path: Path) -> pd.DataFrame:
	"""Load synthetic session IDyOM IC from a CSV export.

	Expected merged_df-style columns include:
	- session_id or melody.name (for example: sessiontune123)
	- note.id
	- cpitch.information.content
	- onset

	Returns a DataFrame with columns: id, idyom_ic, idyom_onset.
	"""
	df = pd.read_csv(session_csv_path)
	required = {"note.id", "cpitch.information.content", "onset"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(
			f"session CSV is missing required columns: {sorted(missing)}"
		)

	if "session_id" in df.columns:
		df = df.assign(id=pd.to_numeric(df["session_id"], errors="coerce"))
	elif "melody.name" in df.columns:
		id_series = (
			df["melody.name"]
			.astype(str)
			.str.extract(r"sessiontune(?P<id>\d+)")["id"]
		)
		df = df.assign(id=pd.to_numeric(id_series, errors="coerce"))
	else:
		raise ValueError("session CSV must contain either 'session_id' or 'melody.name'")

	df["cpitch.information.content"] = pd.to_numeric(
		df["cpitch.information.content"],
		errors="coerce",
	)
	df["note.id"] = pd.to_numeric(df["note.id"], errors="coerce")
	df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
	df = df.dropna(subset=["id", "note.id", "cpitch.information.content", "onset"]).copy()
	if df.empty:
		raise ValueError(f"No usable session rows found in {session_csv_path}")

	df["id"] = df["id"].astype(int)
	df = df.sort_values(["id", "note.id", "onset"])

	out = df.groupby("id", as_index=False).apply(
		lambda g: pd.Series(
			{
				"idyom_ic": np.asarray(g["cpitch.information.content"].to_numpy(dtype=float), dtype=float),
				"idyom_onset": np.asarray(g["onset"].to_numpy(dtype=float), dtype=float),
			}
		)
	)
	return out.sort_values("id").reset_index(drop=True)

def load_idyom_session_ic_from_dat_dir(
	dat_dir: Path,
	cache_dir: Path | None = None,
	num_proc: int | None = 4,
	instruments: list[str]  = ['Pad 1 (new age)', 'Synth Voice','Acoustic Guitar (nylon)', 'Acoustic Grand Piano','Trumpet'],
	n_midis: int | None = None,
) -> pd.DataFrame:
	"""Load session synthetic IDyOM using the notebook's merged_df workflow.

	This follows notebooks/ismir25_iccorr.ipynb:
	1) concat all *.dat into df_idyom_session
	2) derive session_id from melody.name
	3) build session_onsets from SynthDataset midi files
	4) merged_ = merge(df_idyom_session[['session_id','note.id','cpitch.information.content']],
	                 session_onsets, on=['session_id','note.id'])
	5) merged_df = merged_.set_index(['session_id','note.id'])

	Returns per-session rows with: id, idyom_ic, idyom_onset.
	"""
 
 
	dat_paths = [str(p) for p in sorted(dat_dir.glob("*.dat"))]

	def _iter_dat_rows(paths):
		for p in paths:
			yield from pd.read_csv(p)[['melody.name', 'note.id', 'cpitch.information.content']].to_dict("records")

	ds_idyom_session = Dataset.from_generator(
    	_iter_dat_rows,
    	gen_kwargs={"paths": dat_paths},
    	cache_dir=cache_dir,
    	num_proc=num_proc,

	)
	ds_idyom_session = ds_idyom_session.map(lambda x: {"session_id": int(x[len("sessiontune"):])}, num_proc=num_proc, input_columns=["melody.name"])
	# ds_idyom_session = ds_idyom_session.filter(lambda session_id: session_id in midi_ids, num_proc=num_proc, input_columns=["session_id"])
	# df_idyom_session = ds_idyom_session.to_pandas()
	df_idyom_session = (
		ds_idyom_session.to_pandas().sort_values(["session_id", "note.id"])        # optional: control order inside array
		.groupby("session_id", as_index=False)
		.agg(
      		idyom_ic=("cpitch.information.content", list),
        	# note_ids=("note.id", list)
		)           # or np.array
	)
	ids = df_idyom_session.session_id.unique()
	ids = ids[:n_midis]
	df_idyom_session = df_idyom_session[df_idyom_session.session_id.isin(ids)]
	midi_files = [Path("/home/mathias/devel/python3/DSTM/data/session/sessiontune{}.mid".format(id_)) for id_ in ids]
	assert [p.exists() for p in midi_files], f"Some MIDI files not found for session IDs: {ids}"
	# midi_path = Path("/home/mathias/devel/python3/DSTM/data/session/")
	# midi_files = np.array(list(midi_path.glob("sessiontune*.mid")))
	# midi_ids = np.array([int(p.stem[len("sessiontune"):]) for p in midi_files])
	# mask = (ids[None] == midi_ids[:, None])
	# hehe = np.argmax(mask, axis=0)
	# midi_files = midi_files[hehe]
	# midi_sorted_idx  = np.argsort(midi_ids)
	# midi_files = midi_files[midi_sorted_idx]
	# midi_ids = midi_ids[midi_sorted_idx]
	# midi_files = midi_files[:n_midis]
	# midi_ids = set(midi_ids[:n_midis])
	
#  .to_pandas()
# 	df_idyom_session["session_id"] = (
# 		df_idyom_session["melody.name"]
# 		.astype(str)
# 		.map(lambda x: int(x[len("sessiontune") :]))
# 	)
	# df_idyom_session = df_idyom_session.sort_values(['session_id', 'note.id'])
	# df_idyom_session[['session_id', 'note.id', 'cpitch.information.content']].pivot('note.id')
	# midi_files = _load_session_midi_files()

	gen_kwargs = dict(
     	midi_files=[str(p) for p in midi_files],
		instruments={'instruments': instruments},
	)
	session_onsets_ds = Dataset.from_generator(_iter_session_note_onsets, gen_kwargs=gen_kwargs, cache_dir=cache_dir, num_proc=num_proc)
	# session_onsets = session_onsets_ds.to_pandas()

	# merged_ = pd.merge(
	# 	df_idyom_session.loc[:, ["session_id", "note.id", "cpitch.information.content"]],
	# 	session_onsets,
	# 	on=["session_id", "note.id"],
	# 	how="inner",
	# )
	# merged_df = merged_.set_index(["session_id", "note.id"]).sort_index()

	# out = []
	# for session_id, g in merged_df.groupby(level=0):
	# 	g = g.reset_index().sort_values("note.id")
	# 	out.append(
	# 		{
	# 			"id": int(session_id),
	# 			"idyom_ic": np.asarray(g["cpitch.information.content"].to_numpy(dtype=float), dtype=float),
	# 			"idyom_onset": np.asarray(g["onset"].to_numpy(dtype=float), dtype=float),
	# 		}
	# 	)

	# if not out:
	# 	raise ValueError(f"No merged session IDyOM/onset rows produced from {dat_dir}")
	# import datasets
	# session_ds = datasets.concatenate_datasets([[ds_idyom_session, session_onsets_ds.remove_columns()],], axis=1)
	concat = session_onsets_ds.add_column("idyom_ic", df_idyom_session["idyom_ic"])
	return concat


def run_idyom_corr_test_for_labels(
	ic_dataset: Dataset | Iterable[Mapping[str, Any]],
 	idyom_df : pd.DataFrame,
	periode_frames: float,
	labels: List[str] = ["ic"],
	combine_inst: bool = True,
    remove_outliers: bool = False,
):
	idyom_all = []
	nlls_all = []
	ic_dataset_df = pd.DataFrame(ic_dataset).sort_values(['session_id','instrument']).set_index('session_id')
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
	return run_idyom_corr_test_for_labels(
		ic_dataset=ic_dataset,
		idyom_df=idyom_df,
		labels=[label],
		sr_idyom=sr_idyom,
		period_music_2_latent=period_music_2_latent,
		idyom_positive_only=idyom_positive_only,
	)


