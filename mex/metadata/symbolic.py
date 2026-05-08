

from dataclasses import dataclass
from io import BytesIO

from datasets.features import Sequence
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value, List as FList


from pathlib import Path
from typing import Callable
from pretty_midi import PrettyMIDI

from ..benchmark import MetaData
from ..dataset_provider import DatasetProvider

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



class IDyOMMetaData(MetaData):
    def __init__(
        self,
        id_path_extractor: Callable[[str], int|str],
        idyom_dat_dir: str = None,
        midi_dir: str = None,
        num_proc: int | None = None,
        instruments: list[str]  = None,
        n_midis: int | None = None
    ):
        super().__init__()
        self.idyom_dat_dir = idyom_dat_dir
        self.midi_dir = midi_dir
        self.id_path_extractor = id_path_extractor
        self.num_proc = num_proc
        self.instruments = instruments
        self.n_midis = n_midis

    @classmethod
    def synth_midi(
        cls,
        midi_data : PrettyMIDI,
        synth_inst: str,
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

    @classmethod
    def _iter_midis(cls,midi_files: list[str], instruments, id_path_extractor):
        """Yield note onset rows for session MIDI files."""
        import pretty_midi
        import soundfile as sf
        instruments = instruments['instruments'] if isinstance(instruments, dict) else instruments
        for midi_path in midi_files:
            p = Path(midi_path)
            id = id_path_extractor(midi_path)
            midi = pretty_midi.PrettyMIDI(str(p))
            if not midi.instruments:
                continue
            audios = {}
            for synth_inst in instruments:
                audio = cls.synth_midi(midi, synth_inst)
                buf = BytesIO()
                sf.write(buf, audio, 44100, format="MP3")
                audios[synth_inst] = buf.getvalue()

            yield {
                    "id": id,
                    "onset": [note.start for instr in midi.instruments for note in instr.notes],
                    "audios": audios,
            }
    @classmethod
    def idyom_ds(
        cls,
        idyom_dat_dir: str,
        midi_dir: str,
        id_path_extractor: Callable[[str], int|str],
        cache_dir: Path | None = None,
        num_proc: int | None = None,
        instruments: list[str]  = ['Pad 1 (new age)', 'Synth Voice','Acoustic Guitar (nylon)', 'Acoustic Grand Piano','Trumpet'],
        n_midis: int | None = None,
    ) -> Dataset:
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
        # TODO: make sure that onsets for entropy is correct
        idyom_dat_dir = Path(idyom_dat_dir)
        dat_paths = [str(p) for p in sorted(idyom_dat_dir.glob("*.dat"))]

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
        session_onsets_ds = Dataset.from_generator(cls._iter_midis, gen_kwargs=gen_kwargs, cache_dir=cache_dir, num_proc=num_proc)
        concat = session_onsets_ds.add_column("idyom_ic", df_idyom_session["idyom_ic"]).add_column("idyom_entropy", df_idyom_session["idyom_entropy"])
        return concat
    def _preprocess(self, cache_dir: str = None) -> Dataset:
        return self.idyom_ds(
            idyom_dat_dir=self.idyom_dat_dir,
            midi_dir=self.midi_dir,
            id_path_extractor=self.id_path_extractor,
            cache_dir=cache_dir,
            num_proc=self.num_proc,
            instruments=self.instruments,
            n_midis=self.n_midis,
        )
    @classmethod
    def parse_metadata_scheme(cls, dataset: Dataset):
        scheme = {'id': Value('int64'), 'onset': FList(Value('float64'))}
        no_missing_collumns = False
        if "idyom_ic" in dataset.features:
            scheme['idyom_ic'] = FList(Value('float64'))
            no_missing_collumns = True
        if "idyom_entropy" in dataset.features:
            scheme['idyom_entropy'] = FList(Value('float64'))
            no_missing_collumns = True
        assert no_missing_collumns, "Some required columns are missing from the dataset"
        dataset.select_columns(list(scheme.keys())).cast(Features(scheme))

class SessionIDyOMMetaData(IDyOMMetaData):
    def __init__(
        self,
        idyom_dat_dir: str = None,
        midi_dir: str = None,
        num_proc: int | None = None,
        instruments: list[str]  = ['Pad 1 (new age)', 'Synth Voice','Acoustic Guitar (nylon)', 'Acoustic Grand Piano','Trumpet'],
        n_midis: int | None = 200,
    ):
        super().__init__(
            idyom_dat_dir=idyom_dat_dir,
            midi_dir=midi_dir,
            id_path_extractor = lambda x: int(Path(x).stem[len("sessiontune"):]),
            num_proc=num_proc,
            instruments=instruments,
            n_midis=n_midis,
        )
    @property
    def ds_name(self) -> str:
        return "session_idyom"
