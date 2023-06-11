from mgt.datamanagers.compound_word.compound_word_mapper import CompoundWordMapper
from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper
from tension_calculation import *

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
        
        def cal_diameter1(piano_roll,key_index: int) -> List[int]:
            diameters = []
            indices = []
            for i in piano_roll:
                shifte = i - key_index
                if shifte < 0:
                    shifte += 12
                indices.append(note_index_to_pitch_index[shifte])
            diameters.append(largest_distance(indices))
            return diameters
        
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
                        n =[0,0,-1,-1,-1,-1,-1,-1]
                        r = 2
                        for j in i:
                            n[0] = j[0]
                            n[1] = j[1]
                            n[r] = j[2]
                            
                            if r >= 7:
                                break
                            r = r + 1
                        p.append(n)
                    if p[-1] == [2, 0, 0, 0, 0, 0, 0, 0]:
                        del p[-1]
                        
                    q1 = []
                    for i in p:
                        s = []
                        if i[0] == 3:
                            if i[2] != -1:
                                s.append(inverse_dic[i[2]][0])
                            if i[3] != -1:
                                s.append(inverse_dic[i[3]][0])
                            if i[4] != -1:
                                s.append(inverse_dic[i[4]][0])
                            if i[5] != -1:
                                s.append(inverse_dic[i[5]][0])
                            if i[6] != -1:
                                s.append(inverse_dic[i[6]][0])
                            if i[7] != -1:
                                s.append(inverse_dic[i[7]][0])
                            q1.append(s)
                        
                    centroids = []
                    for iii in q1:
                        if len(iii)  == 1:
                            centroids.append(5)
                        else:
                            centroids.append(cal_diameter1(iii,0)[0])
                            
                    centroids1 = []
                    for (i,iii) in enumerate(q1):
                        if i < len(q1)-1:
                            centroids1.append(notes_to_ce(q1[i+1],0) - notes_to_ce(q1[i],0))
                    centroids1.append(np.array([5,5,5]))       
                    key_dife = np.linalg.norm(centroids1, axis=-1)
                    
                    pq = []
                    for i in p:
                        r = 0
                        if i[0] == 3:
                            if i[2] != -1:
                                r = r + 1
                            if i[3] != -1:
                                r = r + 1
                            if i[4] != -1:
                                r = r + 1
                            if i[5] != -1:
                                r = r + 1
                            if i[6] != -1:
                                r = r + 1
                            if i[7] != -1:
                                r = r + 1
                        pq.append([i[0],i[1]]+sorted([i[2],i[3],i[4],i[5],i[6],i[7]], reverse=True)+[r])
                        
                    dia = [0,1.46,1.6,1.85,2.15,2.44,3.124,3.136,3.2,3.86,4.47,4.62,5]
                    dife = [0.0, 0.020202020202020193, 0.09090909090909091, 0.09270944570168699, 0.10808628325601345, 0.10808628325601347, 0.10827491302293185, 0.10827491302293187, 0.11013905867083648, 0.1111111111111111, 0.11861296312552, 0.12553813998539817, 0.12957670877434, 0.13173876438169718, 0.14696938456699069, 0.1469693845669907, 0.1538618516324144, 0.1665986255670086, 0.17356110390903678, 0.17461067804945057, 0.1746106780494506, 0.17871186921319038, 0.1819798177519557, 0.18294640678379565, 0.18303342472865783, 0.1854723699099141, 0.18571184369578825, 0.18571184369578828, 0.18878775470493478, 0.18892223261787974, 0.1897871133667952, 0.19365130117373594, 0.19592835323158522, 0.2, 0.20000000000000007, 0.20041931352683534, 0.20203050891044214, 0.20327890704543547, 0.20396078054371142, 0.20608041101101565, 0.20827332469084403, 0.20995626366712955, 0.2135415650406262, 0.21354156504062627, 0.2154065922853802, 0.2196917449744467, 0.21969174497444682, 0.2207724979877897, 0.2208004403689453, 0.22213150075601149, 0.2222222222222222, 0.22268088570756164, 0.22879178091082225, 0.2316635341652468, 0.23570226039551584, 0.23636363636363641, 0.2424242424242424, 0.24348706415993265, 0.24454238537017814, 0.2449489742783178, 0.24494897427831783, 0.25155319983192226, 0.256124969497314, 0.2569911368986184, 0.25699113689861847, 0.25798968523169763, 0.265150641496141, 0.26620330112690976, 0.2711401631305156, 0.27114016313051564, 0.2785677655436824, 0.27856776554368246, 0.2800291530012764, 0.28017487250585743, 0.2939387691339814, 0.29416086117105716, 0.29667022624539835, 0.3073330883274691, 0.3091206165165234, 0.30912061651652345, 0.3091206165165235, 0.30912061651652356, 0.3124099870362662, 0.3253681957566187, 0.32796216805215417, 0.3307305792474937, 0.3315805769527909, 0.3333333333333333, 0.34110136816367415, 0.3452298849598449, 0.35136418446315326, 0.3565391723301883, 0.36490485822410695, 0.3671301955694549, 0.3720082303616281, 0.37434294373819704, 0.3758118783397281, 0.376796110173626, 0.38376947813196277, 0.38678159211627433, 0.38873012632301995, 0.38873012632302006, 0.3904272341431856, 0.3922144815614096, 0.392491511725871, 0.39810663004889646, 0.4, 0.4000000000000001, 0.4102005153701265, 0.41182520563948005, 0.4151453709393202, 0.42237424163885756, 0.42344080109194104, 0.4311919881123082, 0.43555162199158576, 0.44050476097818114, 0.45370133122923906, 0.4568491119736483, 0.46168379594441233, 0.4621688003316538, 0.46427960923947065, 0.47140452079103173, 0.4789444175497204, 0.48332183894378294, 0.5183068350973601, 0.5206833117271102, 0.5206833117271104, 0.5285599057070339, 0.5324066022538195, 0.5337583340648402, 0.5447844743169451, 0.5637178175095922, 0.5878775382679627, 0.5965083588871344, 0.6000000000000001, 0.618241233033047, 0.6204014500173847, 0.6256019323641655, 0.6303030303030304, 0.6308400618805604, 0.6446359868604573, 0.6642548967837832, 0.676280542305356, 0.6816972283287582, 0.6852736679604725, 0.6863753427324668, 0.7071067811865476, 0.7348469228349533, 0.7504268757632342, 0.76, 0.7832428663927313, 0.7867733763155768, 0.7924021924300193, 0.7986099033807293, 0.812403840463596, 0.8158431221748457, 0.8330932987633766, 0.8459051693633013, 0.8541662601625049, 0.8666666666666668, 0.8724875746400257, 0.873053390247253, 0.8816775895421333, 0.9130169768410661, 0.9273618495495703, 0.9285592184789413, 0.9579144011862437, 0.966229786334493, 0.9744742172063867, 0.9787538933120364, 0.992191737742481, 1.0, 1.019803902718557, 1.0225458424931373, 1.0366136701947202, 1.0476640682967036, 1.0677078252031311, 1.077032961426901, 1.0808628325601344, 1.0827491302293186, 1.100187812666772, 1.1125133484202439, 1.127435635019184, 1.1335333957072784, 1.1661903789690602, 1.188949115816148, 1.224744871391589, 1.2308948130593151, 1.3224556283251583, 1.55492050529208, 1.5684387141358123, 1.5972198067614582, 1.6377491328717662, 1.6911534525287764, 1.7152378308347012, 1.7204650534085253, 1.7333333333333336, 1.7786109158901102, 1.8037738217415178, 1.8303342472865778, 1.8402299532916357, 1.84413023578342, 1.873392644375439, 1.8867962264113207, 1.9332873557751316, 1.9905331502444823, 2.0452727191912006, 2.0591260281974, 2.103711006768753, 2.23606797749979, 2.449489742783178,8]
                    
                    pqq =[]
                    n = 0
                    for i in range(len(pq)):
                        if pq[i][0] == 2:
                            pqq.append([1,0,0,0,0,0,0,0,0,0])
                        else:
                            if i+1 >= len(pq):
                                pqq.append([2,pq[i][1],pq[i][2]+1,pq[i][3]+1,pq[i][4]+1,pq[i][5]+1,pq[i][6]+1,pq[i][7]+1,np.argmin(np.abs(np.array(dia) - centroids[n])).item(),np.argmin(np.abs(np.array(dife) - key_dife[n])).item()])
                            else:
                                if pq[i+1][2] == 0:
                                    if i+2 >= len(pq):
                                        pqq.append([2,pq[i][1],pq[i][2]+1,pq[i][3]+1,pq[i][4]+1,pq[i][5]+1,pq[i][6]+1,pq[i][7]+1,np.argmin(np.abs(np.array(dia) - centroids[n])).item(),np.argmin(np.abs(np.array(dife) - key_dife[n])).item()])
                                    else:
                                        pqq.append([2,pq[i][1],pq[i][2]+1,pq[i][3]+1,pq[i][4]+1,pq[i][5]+1,pq[i][6]+1,pq[i][7]+1,np.argmin(np.abs(np.array(dia) - centroids[n])).item(),np.argmin(np.abs(np.array(dife) - key_dife[n])).item()])
                                else:
                                    pqq.append([2,pq[i][1],pq[i][2]+1,pq[i][3]+1,pq[i][4]+1,pq[i][5]+1,pq[i][6]+1,pq[i][7]+1,np.argmin(np.abs(np.array(dia) - centroids[n])).item(),np.argmin(np.abs(np.array(dife) - key_dife[n])).item()])
                                
                            n = n + 1
                    print(f'Extracted {len(pqq)} compound words.')                    
                    training_data.append(pqq)
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
        for i in data:
            n = 0
            if i[0] == 1:
                q.append([2,0,0,0,0,0,0,0])
            else:
                q.append([2,i[1],0,0,0,0,0,0])
                for j in range(6):
                    if i[j+2] != 0:
                        q.append([3,i[1],0,0,*inverse_dic[int(i[j+2]-1)],31])

        remi = self.compound_word_mapper.map_to_remi(q)
        return MidiToolkitWrapper(self.to_midi_mapper.to_midi(remi))
