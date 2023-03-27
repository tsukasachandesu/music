from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper, PrettyMidiWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig
from mgt.datamanagers.remi.efficient_remi_converter import EfficientRemiConverter
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper
from mgt.datamanagers.a import *
import numpy as np
from pretty_midi import PrettyMIDI

defaults = {
    'use_chords': True,
    'use_note_name': True,
    'transposition_steps': [0],
    'map_tracks_to_instruments': {},
    'instrument_mapping': {},
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

                        def to_midi1(data) -> MidiWrapper:
                            if self.efficient_remi_config.enabled:
                                efficient_words = list(map(lambda x: self.dictionary.data_to_word(x), data))
                                words = self.efficient_remi_converter.convert_to_normal_remi(efficient_words)
                                data = self.data_extractor.words_to_data(words)
                            return MidiToolkitWrapper(self.to_midi_mapper.to_midi(data))   
                        midi = to_midi1(data)
                        midi.save("a.midi")
                        pm = pretty_midi.PrettyMIDI("a.midi")
                        pm = remove_drum_track(pm)
                        sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = get_beat_time(pm, beat_division=16)
                        piano_roll = get_piano_roll(pm, sixteenth_time)

                        key_name = all_key_names
                        key_name, key_pos, note_shift = cal_key(piano_roll, key_name, end_ratio=0.5)
                        centroids = cal_centroid(piano_roll, note_shift, -1, -1)
                        print(centroids)

                        window_size = 1
                        aho = 1
                        if aho == 1:
                            # use a bar window to detect key change
                            merged_centroids = merge_tension(
                                centroids, beat_indices, down_beat_indices, window_size=-1)

                            silent = np.where(np.linalg.norm(merged_centroids, axis=-1) == 0)
                            merged_centroids = np.array(merged_centroids)

                            key_diff = merged_centroids - key_pos
                            key_diff = np.linalg.norm(key_diff, axis=-1)

                            key_diff[silent] = 0

                            diameters = cal_diameter(piano_roll, note_shift, -1, -1)
                            print(diameters)
                            diameters = merge_tension(
                                diameters, beat_indices, down_beat_indices, window_size=-1)
                            #

                            key_change_bar = detect_key_change(
                                key_diff, diameters, start_ratio=0.5)
                            if key_change_bar != -1:
                                key_change_beat = np.argwhere(
                                    beat_time == down_beat_time[key_change_bar])[0][0]
                                change_time = down_beat_time[key_change_bar]
                                changed_key_name, changed_key_pos, changed_note_shift = get_key_index_change(
                                    pm, change_time, sixteenth_time)
                                if changed_key_name != key_name:
                                    m = int(change_time // 60)
                                    s = int(change_time % 60)

                                else:
                                    changed_note_shift = -1
                                    changed_key_name = ''
                                    key_change_beat = -1
                                    change_time = -1
                                    key_change_bar = -1

                            else:
                                changed_note_shift = -1
                                changed_key_name = ''
                                key_change_beat = -1
                                change_time = -1

                        else:
                            changed_note_shift = -1
                            changed_key_name = ''
                            key_change_beat = -1
                            change_time = -1
                            key_change_bar = -1

                        centroids = cal_centroid(
                            piano_roll, note_shift, key_change_beat, changed_note_shift)

                        merged_centroids = merge_tension(
                            centroids, beat_indices, down_beat_indices, window_size=window_size)
                        merged_centroids = np.array(merged_centroids)

                        silent = np.where(np.linalg.norm(merged_centroids, axis=-1) < 0.1)

                        if window_size == -1:
                            window_time = down_beat_time
                        else:
                            window_time = beat_time[::window_size]

                        if key_change_beat != -1:
                            key_diff = np.zeros(merged_centroids.shape[0])
                            changed_step = int(key_change_beat / abs(window_size))
                            for step in range(merged_centroids.shape[0]):
                                if step < changed_step:
                                    key_diff[step] = np.linalg.norm(
                                        merged_centroids[step] - key_pos)
                                else:
                                    key_diff[step] = np.linalg.norm(
                                        merged_centroids[step] - changed_key_pos)
                        else:
                            key_diff = np.linalg.norm(merged_centroids - key_pos, axis=-1)
                        key_diff[silent] = 0

                        diameters = cal_diameter(
                            piano_roll, note_shift, key_change_beat, changed_note_shift)
                        diameters = merge_tension(
                            diameters, beat_indices, down_beat_indices, window_size)
                        #
                        diameters[silent] = 0

                        centroid_diff = np.diff(merged_centroids, axis=0)
                        #
                        np.nan_to_num(centroid_diff, copy=False)

                        centroid_diff = np.linalg.norm(centroid_diff, axis=-1)
                        centroid_diff = np.insert(centroid_diff, 0, 0)

                        total_tension = key_diff
                        print(total_tension.size)
                        print(diameters.size)
                        print(centroid_diff.size)
                        
                        total1 = [0,0.2,0.4,0.6,0.8,1,1.2,1.4]
                        diamet1 = [0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,3.9,4.2]
                        centroid1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6]
                        print(total_tension.size)

                        numi = 0
                        for (i,j) in enumerate(words):
                          if "Bar" in j:
                            numi=numi+1
                        print(numi)
                        numin = 0
                        for (i,j) in enumerate(words):
                          if "Bar" in j: 
                            words[i+1] = "total_" +  str(np.argmin(np.abs(np.array(total1) - total_tension[numin])))
                            words[i+2] = "diamet_" +  str(np.argmin(np.abs(np.array(diamet1) - diameters[numin]))) 
                            words[i+3] = "centroid_" +  str(np.argmin(np.abs(np.array(centroid1) - centroid_diff[numin])))                            
                            numin = numin + 1

                        data = self.data_extractor.words_to_data(words)
                        print(data)

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
