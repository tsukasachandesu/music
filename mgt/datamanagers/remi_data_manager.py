from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig
from mgt.datamanagers.remi.efficient_remi_converter import EfficientRemiConverter
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper

import numpy as np

defaults = {
    'use_chords': False,
    'use_note_name': True,
    'transposition_steps':[0],
    'map_tracks_to_instruments': {},
    'instrument_mapping': {
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
        15:0,
        16:0,
        17:0,
        18:0,
        19:0,
        20:0,
        21:0,
        22:0,
        23:0,
        24:0,
        25:0,
        26:0,
        27:0,
        28:0,
        29:0,
        30:0,
        31:0,
        32:None,
        33:None,
        34:None,
        35:None,
        36:None,
        37:None,
        38:None,
        39:None,
        40:0,
        41:0,
        42:0,
        43:0,
        44:0,
        45:0,
        46:0,
        47:0,
        48:0,
        49:0,
        50:0,
        51:0,
        52:0,
        53:0,
        54:0,
        55:0,
        56:0,
        57:0,
        58:0,
        59:0,
        60:0,
        61:0,
        62:0,
        63:0,
        64:0,
        65:0,
        66:0,
        67:0,
        68:0,
        69:0,
        70:0,
        71:0,
        72:0,
        73:0,
        74:0,
        75:0,
        76:0,
        77:0,
        78:0,
        79:0,
        80:0,
        81:0,
        82:0,
        83:0,
        84:0,
        85:0,
        86:0,
        87:0,
        88:0,
        89:0,
        90:0,
        91:0,
        92:0,
        93:0,
        94:0,
        95:0,
        96:0,
        97:0,
        98:0,
        99:0,
        100:0,
        101:0,
        102:0,
        103:0,
        104:0,
        105:0,
        106:0,
        107:0,
        108:0,
        109:0,
        110:0,
        111:0,
        112:None,
        113:None,
        114:None,
        115:None,
        116:None,
        117:None,
        118:None,
        119:None,
        120:None,
        121:None,
        122:None,
        123:None,
        124:None,
        125:None,
        126:None,
        127:None
    },
    'efficient_remi_config': EfficientRemiConfig()
}


class RemiDataManager(DataManager):
    """
    use_chords: Should the data manager try to extract chord events based on the played notes.
                This does not work very well for multi instrument midi.
    transposition_steps: Transposed copies of the data to include. For example [-1, 0, 1] has a copy that is transposed
                One semitone down, once the original track, and once transposed one semitone up.
    map_tracks_to_instruments: Whether to map certain track numbers to instruments. For example {0=0, 1=25} maps
                track 0 to a grand piano, and track 1 to an acoustic guitar.
    instrument_mapping: Maps instruments to different instruments. For example {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
                maps all piano-like instruments to a grand piano. Mapping to None removes the instrument entirely.
    efficient_remi: Does not repeat instrument and position for every note if they are the same as the previous.
    """

    def __init__(
            self,
            use_chords=defaults['use_chords'],
            use_note_name=defaults['use_note_name'],
            transposition_steps=defaults['transposition_steps'],
            map_tracks_to_instruments=defaults['map_tracks_to_instruments'],
            instrument_mapping=defaults['instrument_mapping'],
            efficient_remi_config=defaults['efficient_remi_config']
    ):
        self.use_chords = use_chords
        self.use_note_name = use_note_name
        self.transposition_steps = transposition_steps
        self.map_tracks_to_instruments = map_tracks_to_instruments
        self.instrument_mapping = instrument_mapping
        self.dictionary = DictionaryGenerator.create_dictionary()
        self.data_extractor = DataExtractor(
            dictionary=self.dictionary,
            map_tracks_to_instruments=self.map_tracks_to_instruments,
            use_chords=self.use_chords,
            use_note_name=self.use_note_name,
            instrument_mapping=self.instrument_mapping
        )
        self.efficient_remi_config = efficient_remi_config
        if self.efficient_remi_config.enabled:
            self.efficient_remi_converter = EfficientRemiConverter(efficient_remi_config)
        self.to_midi_mapper = ToMidiMapper(self.dictionary)

    def prepare_data(self, midi_paths) -> DataSet:

        training_data = []
        for path in midi_paths:
            for transposition_step in self.transposition_steps:
                try:

                    if self.efficient_remi_config.enabled:
                        events = self.data_extractor.extract_events(path, transposition_step)
                        words = self.efficient_remi_converter.convert_to_efficient_remi(events)

                        data = self.data_extractor.words_to_data(words)
                        print(f"Parsed {len(data)} words from midi as efficient REMI.")

                        training_data.append(data)
                    else:
                        data = self.data_extractor.extract_data(path, transposition_step)
                        training_data.append(data)
                except Exception as e:
                    print(e)

        return DataSet(training_data, self.dictionary)

    def to_midi(self, data) -> MidiWrapper:
        if self.efficient_remi_config.enabled:
            efficient_words = list(map(lambda x: self.dictionary.data_to_word(x), data))
            words = self.efficient_remi_converter.convert_to_normal_remi(efficient_words)
            data = self.data_extractor.words_to_data(words)

        return MidiToolkitWrapper(self.to_midi_mapper.to_midi(data))
