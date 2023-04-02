import string

from mgt.datamanagers.data_manager import Dictionary

NUMBER_OF_TEMPO_VALUES = 64


class WordType(object):
    PADDING = 0
    EOS = 1
    TIMING = 2
    NOTE = 3


class CompoundWord(object):
    word_type: int
    bar_beat: int
    instrument: int
    note_name: int
    octave: int
    duration: int
    velocity: int

    def __init__(self, word_type, bar_beat=0, instrument=0, note_name=0, octave=0, duration=0, velocity=0):
        self.word_type = word_type
        self.bar_beat = bar_beat
        self.instrument = instrument
        self.note_name = note_name
        self.octave = octave
        self.duration = duration
        self.velocity = velocity

    def __repr__(self):
        type_string = 'UNKNOWN'
        if self.word_type == 0:
            type_string = 'PADDING'
        elif self.word_type == 1:
            type_string = 'EOS'
        elif self.word_type == 2:
            type_string = 'TIMING'
        elif self.word_type == 3:
            type_string = 'NOTE'

        return f'CompoundWord(type={type_string}, bar_beat={self.bar_beat}, instrument={self.instrument}, note_name={self.note_name}, octave={self.octave}, duration={self.duration}, velocity={self.velocity})'


def create_bar_event():
    return CompoundWord(word_type=WordType.TIMING, bar_beat=0)


def create_beat_event(beat):
    return CompoundWord(word_type=WordType.TIMING, bar_beat=beat + 1)


def create_note_event(instrument, note_name, octave, duration, velocity):
    return CompoundWord(
        word_type=WordType.NOTE,
        instrument=instrument,
        note_name=note_name,
        octave=octave,
        duration=duration,
        velocity=velocity)


def create_eos_event():
    return CompoundWord(word_type=WordType.EOS)


def map_word(key, offset):
    return key - offset


class CompoundWordMapper(object):

    def __init__(self, dictionary):
        self.dictionary = dictionary

        instrument_keys = {k: v for k, v in dictionary.dtw.items() if 'Instrument' in v}.keys()
        self.instrument_size = len(instrument_keys)
        self.instrument_offset = min(instrument_keys)


        note_duration_keys = {k: v for k, v in dictionary.dtw.items() if 'Note Duration' in v}.keys()
        self.note_duration_size = len(note_duration_keys)
        self.note_duration_offset = min(note_duration_keys)

        note_velocity_keys = {k: v for k, v in dictionary.dtw.items() if 'Note Velocity' in v}.keys()
        self.note_velocity_size = len(note_velocity_keys)
        self.note_velocity_offset = min(note_velocity_keys)

        position_keys = {k: v for k, v in dictionary.dtw.items() if 'Position' in v}.keys()
        self.position_size = len(position_keys)
        self.position_offset = min(position_keys)

        note_name_keys = {k: v for k, v in dictionary.dtw.items() if 'Note Name' in v}.keys()
        self.note_name_size = len(note_name_keys)
        self.note_name_offset = min(note_name_keys)

        octave_keys = {k: v for k, v in dictionary.dtw.items() if 'Note Octave' in v}.keys()
        self.octave_size = len(octave_keys)
        self.octave_offset = min(octave_keys)

    def map_to_compound(self, remi_words: [string], dictionary: Dictionary) -> [CompoundWord]:
        compound_words = []
        prev_position = None
        for i in range(len(remi_words)):
            if remi_words[i] == 'Bar_None':
                compound_words.append(create_bar_event())
                prev_position = None
            elif i + 5 < len(remi_words) and \
                    'Position' in remi_words[i] and \
                    'Instrument' in remi_words[i + 1] and \
                    'Note Velocity' in remi_words[i + 2] and \
                    'Note Name' in remi_words[i + 3] and \
                    'Note Octave' in remi_words[i + 4] and \
                    'Note Duration' in remi_words[i + 5]:

                current_position = map_word(dictionary.wtd[remi_words[i]], self.position_offset)
                if prev_position is None or prev_position != current_position:
                    compound_words.append(create_beat_event(current_position))
                    prev_position = current_position

                instrument_position = map_word(dictionary.wtd[remi_words[i + 1]], self.instrument_offset)
                velocity_position = map_word(dictionary.wtd[remi_words[i + 2]], self.note_velocity_offset)
                note_name_position = map_word(dictionary.wtd[remi_words[i + 3]], self.note_name_offset)
                octave_position = map_word(dictionary.wtd[remi_words[i + 4]], self.octave_offset)
                duration_position = map_word(dictionary.wtd[remi_words[i + 5]], self.note_duration_offset)
                compound_words.append(create_note_event(
                    instrument=instrument_position,
                    velocity=velocity_position,
                    note_name=note_name_position,
                    octave=octave_position,
                    duration=duration_position))
            elif i + 2 < len(remi_words) and \
                    'Position' in remi_words[i] and \
                    'Tempo Class' in remi_words[i + 1] and \
                    'Tempo Value' in remi_words[i + 2]:

                current_position = map_word(dictionary.wtd[remi_words[i]], self.position_offset)
                if prev_position is None or prev_position != current_position:
                    compound_words.append(create_beat_event(current_position))
                    prev_position = current_position

        compound_words.append(create_eos_event())

        return compound_words

    @staticmethod
    def map_compound_words_to_data(compound_words: [CompoundWord]):
        return list(map(lambda x: [
            x.word_type,
            x.bar_beat,
            x.instrument,
            x.note_name,
            x.octave,
            x.duration,
            x.velocity
        ], compound_words))

    def map_to_remi(self, compound_data: [[int]]):
        result = []
        current_position = 0
        for compound_word in compound_data:
            remi_sequence, current_position = self.map_compound_word_to_remi(compound_word, current_position)
            result.extend(remi_sequence)
        return result

    def map_compound_word_to_remi(self, compound_word, current_position):
        word_type = compound_word[0]
        if word_type == WordType.NOTE:
            return self.map_compound_note_to_remi(compound_word, current_position), current_position
        elif word_type == WordType.TIMING:
            return self.map_compound_timing_to_remi(compound_word)
        else:
            return [], current_position

    def map_compound_note_to_remi(self, compound_word, current_position):
        position = current_position + self.position_offset
        instrument = compound_word[2] + self.instrument_offset
        note_name = compound_word[3] + self.note_name_offset
        octave = compound_word[4] + self.octave_offset
        duration = compound_word[5] + self.note_duration_offset
        velocity = compound_word[6] + self.note_velocity_offset
        return [position, instrument, velocity, note_name, octave, duration]

    def map_compound_timing_to_remi(self, compound_word):
        bar_beat = compound_word[1]
        if bar_beat == 0:
            return [self.dictionary.word_to_data('Bar_None')], 0
        else:
            position = bar_beat - 1 + self.position_offset
            return [position], bar_beat - 1
