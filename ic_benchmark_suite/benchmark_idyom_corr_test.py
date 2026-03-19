import math
import re
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np
import pandas as pd
import pretty_midi
import scipy.signal
import scipy.stats
from datasets import Dataset
from datasets import load_from_disk


def _salami_id_extractor(path: Path) -> int:
	match = re.match(r"^\D*(?P<id>\d+)$", path.stem)
	if match is None:
		raise ValueError(f"Could not parse SALAMI id from filename: {path}")
	return int(match.group("id"))


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


def _load_session_midi_files(
	midi_path: Path = Path("/home/mathias/devel/python3/DSTM/data/session/"),
	session_split_path: Path = Path("session_split.json"),
	n_midis: int | None = None,
) -> list[Path]:
	"""Load session MIDI file paths from the test split, matching notebook setup."""
	split = sorted(json.load(open(session_split_path))["test"])
	midi_files = [Path(midi_path, f"sessiontune{id_}.mid") for id_ in split]
	midi_files = [p for p in midi_files if p.exists()]
	if n_midis is not None:
		midi_files = midi_files[:n_midis]
	return midi_files


def _iter_session_note_onsets(midi_files: list[str]):
	"""Yield note onset rows for session MIDI files."""
	for midi_path in midi_files:
		p = Path(midi_path)
		match = re.match(r"^sessiontune(?P<id>\d+)$", p.stem)
		if match is None:
			continue
		session_id = int(match.group("id"))
		midi = pretty_midi.PrettyMIDI(str(p))
		if not midi.instruments:
			continue
		for note_id, note in enumerate(midi.instruments[0].notes, 1):
			yield {
				"session_id": session_id,
				"note.id": note_id,
				"onset": float(note.start),
			}
		# yield {'session_id': session_id, 'onsets': [float(note.start) for note in midi.instruments[0].notes]}


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
	# dfs = []
	# for p in sorted(dat_dir.glob("*.dat")):
	# 	df = pd.read_csv(p)
	# 	dfs.append(df)
	# if not dfs:
	# 	raise ValueError(f"No .dat files found in {dat_dir}")

	# df_idyom_session = pd.concat(dfs, ignore_index=True)
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
	df_idyom_session = ds_idyom_session.to_pandas()
#  .to_pandas()
# 	df_idyom_session["session_id"] = (
# 		df_idyom_session["melody.name"]
# 		.astype(str)
# 		.map(lambda x: int(x[len("sessiontune") :]))
# 	)
	# df_idyom_session = df_idyom_session.sort_values(['session_id', 'note.id'])
	# df_idyom_session[['session_id', 'note.id', 'cpitch.information.content']].pivot('note.id')
	midi_files = _load_session_midi_files()
	session_onsets_ds = Dataset.from_generator(_iter_session_note_onsets, gen_kwargs={"midi_files": [str(p) for p in midi_files]}, cache_dir=cache_dir, num_proc=num_proc)
	session_onsets = session_onsets_ds.to_pandas()

	merged_ = pd.merge(
		df_idyom_session.loc[:, ["session_id", "note.id", "cpitch.information.content"]],
		session_onsets,
		on=["session_id", "note.id"],
		how="inner",
	)
	merged_df = merged_.set_index(["session_id", "note.id"]).sort_index()

	out = []
	for session_id, g in merged_df.groupby(level=0):
		g = g.reset_index().sort_values("note.id")
		out.append(
			{
				"id": int(session_id),
				"idyom_ic": np.asarray(g["cpitch.information.content"].to_numpy(dtype=float), dtype=float),
				"idyom_onset": np.asarray(g["onset"].to_numpy(dtype=float), dtype=float),
			}
		)

	if not out:
		raise ValueError(f"No merged session IDyOM/onset rows produced from {dat_dir}")

	return pd.DataFrame(out).sort_values("id").reset_index(drop=True)


def _safe_1d_array(values: Any, label: str, id_: int) -> np.ndarray:
	arr = np.asarray(values, dtype=float)
	if arr.ndim != 1:
		raise ValueError(
			f"Expected a 1D sequence for label '{label}' and id={id_}, got shape={arr.shape}"
		)
	return arr


def _fill_nans_with_mean(arr: np.ndarray) -> np.ndarray:
	arr = arr.astype(float, copy=True)
	nan_mask = np.isnan(arr)
	if not nan_mask.any():
		return arr
	mean_val = np.nanmean(arr)
	if np.isnan(mean_val):
		return arr
	arr[nan_mask] = mean_val
	return arr


def _align_our_series_to_idyom(
	our_series: np.ndarray,
	idyom_series: np.ndarray,
	sr_idyom: float,
	period_music_2_latent: float,
	idyom_onset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Align our series to IDyOM resolution and truncate both to common support."""
	if idyom_onset is not None:
		time = np.arange(len(our_series), dtype=float) * period_music_2_latent
		idx = np.searchsorted(time, idyom_onset)
		idx = np.clip(idx, 0, max(len(our_series) - 1, 0))
		valid = np.isfinite(idyom_series) & np.isfinite(idyom_onset)
		idx = idx[valid]
		if len(idx) == 0:
			return np.asarray([], dtype=float), np.asarray([], dtype=float)
		return idyom_series[valid], our_series[idx]

	max_idyom_len = math.floor(len(our_series) * sr_idyom * period_music_2_latent)
	target_len = min(len(idyom_series), max_idyom_len)
	if target_len <= 0:
		return np.asarray([], dtype=float), np.asarray([], dtype=float)

	idyom_trimmed = idyom_series[:target_len]
	our_resampled = scipy.signal.resample(our_series, target_len)
	return idyom_trimmed, our_resampled


def run_idyom_corr_test_for_labels(
	ic_dataset: Dataset | Iterable[Mapping[str, Any]],
	idyom_df: pd.DataFrame,
	labels: List[str],
	sr_idyom: float = 64.0,
	period_music_2_latent: float = 0.093,
	idyom_positive_only: bool = True,
):
	"""Compute per-track and aggregated IDyOM correlations for all labels.

	Args:
		ic_dataset: HF dataset (or iterable of row dicts) containing at least
			`id` and one or more label columns (for example: IC_0.5, nll).
		idyom_df: DataFrame with columns `id` and `idyom_ic`.
		labels: IC/NLL columns in `ic_dataset` to evaluate.
		sr_idyom: IDyOM temporal rate in Hz.
		period_music_2_latent: frame period for our model output.
		idyom_positive_only: if True, apply mask idyom_ic > 0 before corr.

	Returns:
		dict with:
			- per_track: one row per (id, label)
			- per_label_summary: aggregated per-label stats over tracks
			- per_label_global: per-label corr on concatenated aligned samples
	"""
	if not labels:
		raise ValueError("labels must not be empty")

	rows_iter = ic_dataset
	if isinstance(ic_dataset, Dataset):
		available_columns = set(ic_dataset.column_names)
	else:
		rows_iter = list(ic_dataset)
		available_columns = set(rows_iter[0].keys()) if rows_iter else set()

	missing_cols = sorted([label for label in labels if label not in available_columns])
	if missing_cols:
		raise ValueError(f"Missing required columns in ic_dataset: {missing_cols}")
	if "id" not in available_columns:
		raise ValueError("ic_dataset must contain an 'id' column")

	required_idyom_cols = {"id", "idyom_ic"}
	missing_idyom_cols = required_idyom_cols - set(idyom_df.columns)
	if missing_idyom_cols:
		raise ValueError(f"idyom_df is missing required columns: {sorted(missing_idyom_cols)}")

	idyom_lookup = {}
	for _, row in idyom_df.iterrows():
		entry = {"idyom_ic": np.asarray(row["idyom_ic"], dtype=float), "idyom_onset": None}
		if "idyom_onset" in idyom_df.columns and row["idyom_onset"] is not None:
			entry["idyom_onset"] = np.asarray(row["idyom_onset"], dtype=float)
		idyom_lookup[int(row["id"])] = entry

	per_track_rows = []
	by_label_pairs: dict[str, dict[str, list[np.ndarray]]] = {
		label: {"idyom": [], "ours": []} for label in labels
	}

	for ex in rows_iter:
		id_ = int(ex["id"])
		idyom_entry = idyom_lookup.get(id_)
		if idyom_entry is None:
			continue

		idyom_series = np.asarray(idyom_entry["idyom_ic"], dtype=float)
		idyom_onset = idyom_entry.get("idyom_onset")
		for label in labels:
			our_series = _safe_1d_array(ex[label], label=label, id_=id_)
			our_series = _fill_nans_with_mean(our_series)

			idyom_aligned, our_aligned = _align_our_series_to_idyom(
				our_series=our_series,
				idyom_series=idyom_series,
				sr_idyom=sr_idyom,
				period_music_2_latent=period_music_2_latent,
				idyom_onset=idyom_onset,
			)

			if len(idyom_aligned) == 0:
				per_track_rows.append(
					{
						"id": id_,
						"label": label,
						"n": 0,
						"pearson_r": np.nan,
						"pearson_p": np.nan,
						"spearman_r": np.nan,
						"spearman_p": np.nan,
					}
				)
				continue

			mask = np.isfinite(idyom_aligned) & np.isfinite(our_aligned)
			if idyom_positive_only:
				mask = mask & (idyom_aligned > 0.0)

			x = idyom_aligned[mask]
			y = our_aligned[mask]

			n = len(x)
			if n < 2:
				per_track_rows.append(
					{
						"id": id_,
						"label": label,
						"n": n,
						"pearson_r": np.nan,
						"pearson_p": np.nan,
						"spearman_r": np.nan,
						"spearman_p": np.nan,
					}
				)
				continue

			pear = scipy.stats.pearsonr(x, y)
			spear = scipy.stats.spearmanr(x, y)
			per_track_rows.append(
				{
					"id": id_,
					"label": label,
					"n": n,
					"pearson_r": float(pear.statistic),
					"pearson_p": float(pear.pvalue),
					"spearman_r": float(spear.statistic),
					"spearman_p": float(spear.pvalue),
				}
			)

			by_label_pairs[label]["idyom"].append(x)
			by_label_pairs[label]["ours"].append(y)

	per_track_df = pd.DataFrame(per_track_rows)
	if per_track_df.empty:
		empty = pd.DataFrame()
		return {
			"per_track": empty,
			"per_label_summary": empty,
			"per_label_global": empty,
		}

	per_label_summary = (
		per_track_df.groupby("label", as_index=False)
		.agg(
			n_tracks=("id", "nunique"),
			mean_n=("n", "mean"),
			median_n=("n", "median"),
			mean_pearson_r=("pearson_r", "mean"),
			median_pearson_r=("pearson_r", "median"),
			mean_spearman_r=("spearman_r", "mean"),
			median_spearman_r=("spearman_r", "median"),
		)
		.sort_values("label")
		.reset_index(drop=True)
	)

	global_rows = []
	for label in labels:
		idyom_chunks = by_label_pairs[label]["idyom"]
		our_chunks = by_label_pairs[label]["ours"]
		if not idyom_chunks:
			global_rows.append(
				{
					"label": label,
					"n": 0,
					"pearson_r": np.nan,
					"pearson_p": np.nan,
					"spearman_r": np.nan,
					"spearman_p": np.nan,
				}
			)
			continue

		x = np.concatenate(idyom_chunks)
		y = np.concatenate(our_chunks)
		if len(x) < 2:
			global_rows.append(
				{
					"label": label,
					"n": len(x),
					"pearson_r": np.nan,
					"pearson_p": np.nan,
					"spearman_r": np.nan,
					"spearman_p": np.nan,
				}
			)
			continue

		pear = scipy.stats.pearsonr(x, y)
		spear = scipy.stats.spearmanr(x, y)
		global_rows.append(
			{
				"label": label,
				"n": len(x),
				"pearson_r": float(pear.statistic),
				"pearson_p": float(pear.pvalue),
				"spearman_r": float(spear.statistic),
				"spearman_p": float(spear.pvalue),
			}
		)

	per_label_global = pd.DataFrame(global_rows).sort_values("label").reset_index(drop=True)
	return {
		"per_track": per_track_df,
		"per_label_summary": per_label_summary,
		"per_label_global": per_label_global,
	}


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


def _print_df(title: str, df: pd.DataFrame, max_rows: int) -> None:
	print(f"\n{title}")
	if df.empty:
		print("<empty>")
		return
	print(df.head(max_rows).to_string(index=False))


if __name__ == "__main__":
	labels = ["IC_0.0", "IC_0.3", "IC_0.5"]
	ic_dataset_path = "/home/mathias/devel/python3/audio-it/ic_benchmark_suite/mock_ic_hf_dataset"
	idyom_npy_dir = "/home/mathias/devel/python3/audio-it/output/idyom_pitch_song"
	session_dat_dir = "/home/mathias/devel/python3/audio-it/idyom/session_test_out_full_pitch"
	max_rows = 20

	# ic_dataset = load_from_disk(ic_dataset_path)
	# if not isinstance(ic_dataset, Dataset):
	# 	raise ValueError(
	# 		f"Expected a Hugging Face Dataset at {ic_dataset_path}, got {type(ic_dataset)}"
	# 	)

	# idyom_df = load_idyom_pitch_ic_from_npy(Path(idyom_npy_dir))
	# results = run_idyom_corr_test_for_labels(
	# 	ic_dataset=ic_dataset,
	# 	idyom_df=idyom_df,
	# 	labels=labels,
	# )

	# print("\n=== Pitch Synthetic IDyOM ===")
	# _print_df("Per-label global correlation", results["per_label_global"], max_rows)
	# _print_df("Per-label summary", results["per_label_summary"], max_rows)
	# _print_df("Per-track rows", results["per_track"], max_rows)

	# Additional synthetic IDyOM source from session .dat files (merged_df workflow).
	session_idyom_df = load_idyom_session_ic_from_dat_dir(
     	Path(session_dat_dir),
		cache_dir="/home/mathias/devel/python3/audio-it/ic_benchmark_suite/cache"
	)
	session_results = run_idyom_corr_test_for_labels(
		ic_dataset=ic_dataset,
		idyom_df=session_idyom_df,
		labels=labels,
 num_proc=4,
	)

	print("\n=== Session Synthetic IDyOM ===")
	_print_df("Per-label global correlation", session_results["per_label_global"], max_rows)
	_print_df("Per-label summary", session_results["per_label_summary"], max_rows)
	_print_df("Per-track rows", session_results["per_track"], max_rows)

