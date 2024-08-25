from mgt.datamanagers.compound_word.compound_word_mapper import CompoundWordMapper
from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper


defaults = {
    'transposition_steps': [0],
    'map_tracks_to_instruments': {},
    'instrument_mapping': {}
}


class CompoundWordDataManager(DataManager):
    """
    transposition_steps: Transposed copies of the data to include. For example [-1, 0, 1] has a copy that is transposed
                One semitone down, once the original track, and once transposed one semitone up.
    map_tracks_to_instruments: Whether to map certain track numbers to instruments. For example {0=0, 1=25} maps
                track 0 to a grand piano, and track 1 to an acoustic guitar.
    instrument_mapping: Maps instruments to different instruments. For example {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
                maps all piano-like instruments to a grand piano. Mapping to None removes the instrument entirely.
    """

    def __init__(
            self,
            transposition_steps=defaults['transposition_steps'],
            map_tracks_to_instruments=defaults['map_tracks_to_instruments'],
            instrument_mapping=defaults['instrument_mapping']
    ):
        self.transposition_steps = transposition_steps
        self.map_tracks_to_instruments = map_tracks_to_instruments
        self.instrument_mapping = instrument_mapping
        self.dictionary = DictionaryGenerator.create_dictionary()
        self.compound_word_mapper = CompoundWordMapper(self.dictionary)
        self.data_extractor = DataExtractor(
            dictionary=self.dictionary,
            map_tracks_to_instruments=self.map_tracks_to_instruments,
            use_chords=False,
            use_note_name=True,
            instrument_mapping=self.instrument_mapping
        )
        self.to_midi_mapper = ToMidiMapper(self.dictionary)

    def prepare_data(self, midi_paths) -> DataSet:
        training_data = []
        for path in midi_paths:
            for transposition_step in self.transposition_steps:
                try:
                    data = self.data_extractor.extract_words(path, transposition_step)

                    compound_words = self.compound_word_mapper.map_to_compound(data, self.dictionary)
                    compound_data = self.compound_word_mapper.map_compound_words_to_data(compound_words)

                    print(f'Extracted {len(compound_data)} compound words.')

                    training_data.append(compound_data)
                except Exception as e:
                    print(f"Exception: {e}")

    def prepare_data(self, midi_paths) -> DataSet:
        training_data = []
        for path in midi_paths:
            for transposition_step in self.transposition_steps:
                try:
                    data = self.data_extractor.extract_words(path, transposition_step)

                    compound_words = self.compound_word_mapper.map_to_compound(data, self.dictionary)
                    compound_data = self.compound_word_mapper.map_compound_words_to_data(compound_words)
                    
                    print(f'Extracted {len(compound_data)} compound words.')

                    training_data.append(compound_data)
                except Exception as e:
                    print(f"Exception: {e}")
        
        dataset = DataSet(training_data, self.dictionary)

        for i in range(len(dataset.data)):
            for j in range(len(dataset.data[i])):
                del dataset.data[i][j][2]
                del dataset.data[i][j][-1]
        for i in range(len(dataset.data)):
            bar_offset = 0
            for j in range(len(dataset.data[i])):
                if dataset.data[i][j][0] == 2:
                    bar_offset = dataset.data[i][j][1]
                elif dataset.data[i][j][0] == 3:
                    dataset.data[i][j][1] = bar_offset
        for i in range(len(dataset.data)):
            dataset.data[i] = [
                item for item in dataset.data[i]
                if item[0] != 2 or all(x == 0 for x in item[1:])
            ]
        for i in range(len(dataset.data)):
            count_2 = -1
            for j in range(len(dataset.data[i])):
                if dataset.data[i][j][0] == 2:
                    count_2 += 1
                dataset.data[i][j].append(count_2)
        for i in range(len(dataset.data)):
            dataset.data[i] = [item[1:] for item in dataset.data[i]]
        for i in range(len(dataset.data)):
            for j in range(len(dataset.data[i])):
                if dataset.data[i][j][0] == 0:
                    dataset.data[i][j][0] = 17

        return dataset
    def to_remi(self, data):
        remi = self.compound_word_mapper.map_to_remi(data)
        return list(map(lambda x: self.dictionary.data_to_word(x), remi))

    def to_midi(self, data) -> MidiWrapper:

      def process_dataset(dataset):
        new_data_i = []
        for item in dataset:
            if item[0] != 17:
                new_data_i.append([item[0], 0, 0, 0, 0, 0])
            new_data_i.append(item)
        dataset = new_data_i
        dataset = [
            [2 if all(x == 0 for x in item[2:5]) else 3] + item
            for item in dataset
        ]
        for item in dataset:
            if item[1] == 17:
                item[1] = 0
        dataset = [item[:-1] for item in dataset]
        dataset.data[i] = [item[:2] + [67] + item[2:] for item in dataset]
        dataset.data[i] = [item + [31] for item in dataset]
        return dataset

        
        remi = self.compound_word_mapper.map_to_remi(process_dataset(data))
        return MidiToolkitWrapper(self.to_midi_mapper.to_midi(remi))
        
