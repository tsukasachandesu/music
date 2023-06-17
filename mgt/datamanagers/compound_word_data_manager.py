from mgt.datamanagers.compound_word.compound_word_mapper import CompoundWordMapper
from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper

import numpy as np
import itertools
import math

defaults = {
    'transposition_steps': [0,1,2,3,4,5,6,7,8,-1,-2,-3,-4,-5,-6,-7,-8],
    'map_tracks_to_instruments': {},
    'instrument_mapping': {}
}

def tiv1(q):
    c = [0]*6*2
    c = np.array(c)
    count = 0
    for i in q:
        a = [math.sin(math.radians(30*-i)),math.cos(math.radians(30*-i)),math.sin(math.radians(60*-i)),math.cos(math.radians(60*-i)),math.sin(math.radians(90*-i)),math.cos(math.radians(90*-i)),math.sin(math.radians(120*-i)),math.cos(math.radians(120*-i)),math.sin(math.radians(150*-i)),math.cos(math.radians(150*-i)),math.sin(math.radians(180*-i)),math.cos(math.radians(180*-i))]
        a = np.array(a)
        c = c + a
        count += 1
    if count != 0:
        c /= count
    
    return c.tolist()    

def notes_to_ce(indices):
  note_index_to_pitch_index = [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5]
  total = np.zeros(3)
  count = 0
  for index in indices:
    total += pitch_index_to_position(note_index_to_pitch_index[index])
    count += 1
  if count != 0:
    total /= count               
  return total.tolist()    

def pitch_index_to_position(pitch_index) :
    c = pitch_index - (4 * (pitch_index // 4))
    verticalStep = 0.4
    radius = 1.0
    pos = np.array([0.0, 0.0, 0.0])
    if c == 0:
        pos[1] = radius
    if c == 1:
        pos[0] = radius
    if c == 2:
        pos[1] = -1*radius
    if c == 3:
        pos[0] = -1*radius
    pos[2] = pitch_index * verticalStep
    
    return np.array(pos)

def largest_distance(pitches):
    if len(pitches) < 2:
        return 0
    diameter = 0
    pitch_pairs = itertools.combinations(pitches, 2)
    for pitch_pair in pitch_pairs:
        distance = np.linalg.norm(pitch_index_to_position(
            pitch_pair[0]) - pitch_index_to_position(pitch_pair[1]))
        if distance > diameter:
            diameter = distance
    return diameter

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
        dic = {(i, j, k): index for index, (i, j, k) in enumerate((i, j, k) for j in range(9) for i in range(12) for k in range(64))}
        inverse_dic = {v: k for k, v in dic.items()}

        for path in midi_paths:
            for transposition_step in self.transposition_steps:
                try:
                    data = self.data_extractor.extract_words(path, transposition_step)

                    compound_words = self.compound_word_mapper.map_to_compound(data, self.dictionary)
                    compound_data = self.compound_word_mapper.map_compound_words_to_data(compound_words)
                    a = [[i[0], i[1], dic.get((i[4], i[5], i[6]))] for i in compound_data]
                    d = []
                    for i in a:
                      if i[0] == 2:
                        if i == [2,0,0]:
                          d.append(i)
                        b = i[1]
                      elif i[0] == 3:
                        c = i[2]
                        d.append([3,b,c])
                      else:
                        d.append(i)  
                    cur = 0
                    for i in d:
                        if i == [2, 0, 0]:
                            cur = cur + 1
                    p =[[] * 1 for i in range(cur*16+1)]
                    ppqq =[[i%16+1,6913,6913,6913,6913,6913,6913,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] * 1 for i in range(cur*16+1)]
                    cur = -1
                    for i in d:
                        if i == [2, 0, 0]:
                            cur = cur + 1
                        if i[0] == 3:
                            p[i[1] + cur * 16 -1].append([i[0],i[1],i[2]])

                    pp = []
                    cur = 0
                    for i in p:
                        if cur % 16==0:
                            pp.append([[2, 0, 0]])
                        if i:
                            pp.append(i)
                        cur = cur + 1
                    p  = []
                    for i in pp:
                        n =[0,0,6913,6913,6913,6913,6913,6913]
                        r = 2
                        for j in i:
                            n[0] = j[0]
                            n[1] = j[1]
                            n[r] = j[2]
                            
                            if r >= 7:
                                break
                            r = r + 1
                        p.append(n)
                    if p[-1] == [2, 0, 6913,6913,6913,6913,6913,6913]:
                        del p[-1]
                        
                    q1 = []
                    for i in p:
                        s = []
                        if i[0] == 3:
                            if i[2] != 6913:
                                s.append(inverse_dic[i[2]][0])
                            if i[3] != 6913:
                                s.append(inverse_dic[i[3]][0])
                            if i[4] != 6913:
                                s.append(inverse_dic[i[4]][0])
                            if i[5] != 6913:
                                s.append(inverse_dic[i[5]][0])
                            if i[6] != 6913:
                                s.append(inverse_dic[i[6]][0])
                            if i[7] != 6913:
                                s.append(inverse_dic[i[7]][0])
                            q1.append(s)

                    centroids = []
                    for iii in q1:
                        centroids.append(notes_to_ce(iii))
                        
                    centroids1 = []
                    for iii in q1:
                        centroids1.append(largest_distance(iii))     
                        
                    centroids2 = []
                    for iii in q1:
                        centroids2.append(tiv1(iii))   
                        
                        
                    pq = []
                    for i in p:
                        pq.append([i[0],i[1]]+sorted([i[2],i[3],i[4],i[5],i[6],i[7]]))
                        
                    pqq =[]
                    n = 0
                    for i in range(len(pq)):
                        if pq[i][0] == 2:
                            pqq.append([1,0,0,0,0,0,0,0,0,0,0,0,0])
                        else:
                            pqq.append([2,pq[i][1],pq[i][2]+1,pq[i][3]+1,pq[i][4]+1,pq[i][5]+1,pq[i][6]+1,pq[i][7]+1,centroids[n][0],centroids[n][1],centroids[n][2],centroids1[n], centroids2[n][0],centroids2[n][1],centroids2[n][2],centroids2[n][3],centroids2[n][4],centroids2[n][5],centroids2[n][6],centroids2[n][7],centroids2[n][8],centroids2[n][9],centroids2[n][10],centroids2[n][11] ] )
                            n = n + 1
                            
                    cur = -1
                    for i in range(len(pqq)):
                        if pqq[i][0] == 1:
                            cur  = cur + 1
                        if pqq[i][0] == 2:
                            ppqq[cur*16+pqq[i][1]-1] = [pqq[i][1],pqq[i][2],pqq[i][3],pqq[i][4],pqq[i][5],pqq[i][6],pqq[i][7],pqq[i][8],pqq[i][9],pqq[i][10],pqq[i][11],pqq[i][12],pqq[i][13],pqq[i][14],pqq[i][15],pqq[i][16],pqq[i][17],pqq[i][18],pqq[i][19],pqq[i][20],pqq[i][21],pqq[i][22],pqq[i][23] ] 
                    for i in ppqq:
                        if i[1] == 6914:
                            i[1] = 6913
                        if i[2] == 6914:
                            i[2] = 6913
                        if i[3] == 6914:
                            i[3] = 6913
                        if i[4] == 6914:
                            i[4] = 6913
                        if i[5] == 6914:
                            i[5] = 6913
                        if i[6] == 6914:
                            i[6] = 6913
                            
                    print(f'Extracted {len(ppqq)} compound words.') 
                    
                    training_data.append(ppqq)
                except Exception as e:
                    print(f"Exception: {e}")

        return DataSet(training_data, self.dictionary)

    def to_remi(self, data):
        remi = self.compound_word_mapper.map_to_remi(data)
        return list(map(lambda x: self.dictionary.data_to_word(x), remi))

    def to_midi(self, data) -> MidiWrapper:
        dic = {(i, j, k): index for index, (i, j, k) in enumerate((i, j, k) for j in range(9) for i in range(12) for k in range(64))}
        inverse_dic = {v: k for k, v in dic.items()}
        
        q = []
        n = 0
        for i in data:
            if n %16 == 0:
                q.append([2,0,0,0,0,0,0,0])
            for j in range(6):
                if i[j+1] != 6913:
                    if i[j+1] != 0:
                        q.append([2,n%16+1,0,0,0,0,0,0])
                        q.append([3,0,0,0,*inverse_dic[int(i[j+1]-1)],31])
            n=n+1

        remi = self.compound_word_mapper.map_to_remi(q)
        return MidiToolkitWrapper(self.to_midi_mapper.to_midi(remi))
