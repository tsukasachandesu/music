from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig
from mgt.datamanagers.remi.efficient_remi_converter import EfficientRemiConverter
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper

import numpy as np
from tension_calculation import *

defaults = {
    'use_chords': False,
    'use_note_name': True,
    'transposition_steps':[0],
    'map_tracks_to_instruments': {},
    'instrument_mapping': {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 1,
        10: 0,
        11: 0,
        12: 0,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 2,
        18: 2,
        19: 2,
        20: 2,
        21: 2,
        22: 2,
        23: 2,
        24: 2,
        25: 3,
        26: 3,
        27: 3,
        28: 3,
        29: 3,
        30: 3,
        31: 3,
        32: 3,
        33: 4,
        34: 4,
        35: 4,
        36: 4,
        37: 4,
        38: 4,
        39: 4,
        40: 4,
        41: 5,
        42: 5,
        43: 5,
        44: 5,
        45: 5,
        46: 5,
        47: 5,
        48: 5,
        49: 6,
        50: 6,
        51: 6,
        52: 6,
        53: 6,
        54: 6,
        55: 6,
        56: 6,
        57: 7,
        58: 7,
        59: 7,
        60: 7,
        61: 7,
        62: 7,
        63: 7,
        64: 7,
        65: 8,
        66: 8,
        67: 8,
        68: 8,
        69: 8,
        70: 8,
        71: 8,
        72: 9,
        73: 9,
        74: 9,
        75: 9,
        76: 9,
        77: 9,
        78: 9,
        79: 9,
        80: 9,
        81: 10,
        82: 10,
        83: 10,
        84: 10,
        85: 10,
        86: 10,
        87: 10,
        88: 10},
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
        
        def ce_sum1(indices: List[int], d,start=None, end=None) -> ndarray:
            if not start:
                start = 0
            if not end:
                end = len(indices)
            indices = indices[start:end]
            total = np.zeros(3)
            count = 0
            shift = d
            for timestep, data in enumerate(indices):
                for pitch in data:
                    if pitch:
                        shifted = pitch - shift
                        if shifted < 0:
                            shifted += 12
                        total += pitch_index_to_position(note_index_to_pitch_index[shifted])
                        count += 1
            return total/count
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
        for path in midi_paths:
            for transposition_step in self.transposition_steps:

                    if self.efficient_remi_config.enabled:
                        events = self.data_extractor.extract_events(path, transposition_step)
                        words = self.efficient_remi_converter.convert_to_efficient_remi(events)
                        cur_bar, cur_pos = -1, -1
                        cur = 0
                        for ev in words:
                            if "Bar" in ev:
                                cur += 1
                        poly_record =[[] * 1 for i in range((cur +1)* 16)]
                        pol_record =[[0,0,0,0,0,0,0,0,0,0,0,0] * 1 for i in range((cur +1)* 16)]
                        cur_bar, cur_pos = -1, -1
                        pitch_name_to_pitch_index = {0 :"C", 1:"C#",2:"D",3:"D#",4:"E",5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}
                        pitch_index_to_pitch_name = {v: k for k, v in pitch_name_to_pitch_index.items()}
                        for ev in words:
                            if "Bar" in ev:
                                cur_bar += 1
                            elif "Name" in ev:
                                name = ev.split("_")[1]
                            elif "Position" in ev:
                                posi = ev.split("_")[1].split("/")[0]
                                cur_pos = int(posi)
                            elif "Duration" in ev:
                                posi = ev.split("_")[1]
                                duration = int(posi) + 1
                                st = cur_bar * 16 + cur_pos -1
                                for i in range(duration):
                                    poly_record[st + i].append(pitch_index_to_pitch_name[name])
                        qq=[]
                        for q in poly_record:
                            qq.append(list(set(q)))
                         
                        ahoho = []
                        aho = [0, 7, 2, 9, 4, 11, 5, 10, 3, 8, 1, 6]
                        aho2 = [0, 7, 2, 9, 4, 11, 5, 10, 3, 8, 1, 6,0, 7, 2, 9, 4, 11, 5, 10, 3, 8, 1, 6]
                        for i in aho:
                            ahoho.append( np.linalg.norm(ce_sum1(qq,i)-major_key_position(0)))
                        for i in aho:
                            ahoho.append( np.linalg.norm(ce_sum1(qq,i)-minor_key_position(3)))
                        aaah = ['C major', 'G major', 'D major', 'A major', 'E major', 'B major', 'F major', 'B- major', 'E- major', 'A- major', 'D- major', 'G- major', 'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor', 'D minor', 'G minor', 'C minor', 'F minor', 'B- minor', 'E- minor']
                        anunu= np.argmin(np.array(ahoho))
                        
                        keynub = aho2[anunu]

                        centroids = [np.array([10,10,10])]

                        for (i,iii) in enumerate(qq):
                            if i < len(qq)-1:
                                if (iii == [] ) or (qq[i+1] == []):
                                    centroids.append(np.array([5,5,5]))
                                else:
                                    centroids.append(notes_to_ce(qq[i+1],0)-notes_to_ce(qq[i],0))
                        key_dife = np.linalg.norm(centroids, axis=-1)

                        centroids = []
                        for iii in qq:
                            if iii:
                                centroids.append(notes_to_ce(iii,0))
                            else:
                                centroids.append(np.array([0,0,0]))

                        j= np.empty((1,3))
                        for iii in centroids:
                            if iii.tolist() == [0,0,0]:
                                j = np.append(j,np.array([[2,2,2]]), axis=0)
                            else:
                                j= np.append(j,[iii - major_key_position(note_index_to_pitch_index[keynub])], axis=0)
                        j =np.delete(j, 0, 0)
                        key_diff = np.linalg.norm(j, axis=-1)
                        zz =[]
                        for iiim in qq:
                            if iiim:
                                zz.append(cal_diameter1(iiim,0))
                            else:
                                zz.append([])

                        for (i,j) in enumerate(zz):
                            if j==[]:
                                zz[i] = [5]

                        print(key_dife)
                        print(key_diff)
                        print(zz)

                        b = []
                        numin = 0

                        dia = [0,1.46,1.6,1.85,2.15,2.44,3.124,3.136,3.2,3.86,4.47,4.62,5]
                        cent = [0.0, 0.3000914618978687, 0.32028951652042703, 0.3202895165204271, 0.3223887813189636, 0.3702214073533265, 0.38161462245556216, 0.38318309657916805, 0.387229134827652, 0.3898951952535165, 0.39151051774377665, 0.4053952414996143, 0.4054435663615837, 0.4207579892337162, 0.4236419307670099, 0.44787552456458257, 0.4478755245645826, 0.47505874587740443, 0.47848812472620467, 0.48821356546495104, 0.4882135654649511, 0.4942303072608064, 0.49423030726080647, 0.49825985740775874, 0.5328643230115598, 0.5419660166375292, 0.5492836111736815, 0.5509470110280934, 0.5551739948127485, 0.5557925641756004, 0.5647385904882405, 0.5716142803534566, 0.5992892352986297, 0.6136319915398726, 0.6141925475809684, 0.6288228403265468, 0.6331385989686618, 0.6347866456566333, 0.6389939763648251, 0.6394251211080154, 0.6584338715390867, 0.6629392773882086, 0.6828836544419555, 0.6931239652349445, 0.6948614865597318, 0.7014052149987602, 0.7151221358463669, 0.7364241206696045, 0.7720832115154429, 0.772083211515443, 0.7738932146790165, 0.7916163752121352, 0.8031142418759611, 0.8031142418759614, 0.8073386436335128, 0.8104958954955361, 0.8104958954955362, 0.8153149358914467, 0.8335522449181769, 0.8542310622578778, 0.8625268027742675, 0.8739430676560115, 0.8789924263063933, 0.8866777224353749, 0.8888236572956771, 0.8911352790143595, 0.8958161691780241, 0.9039701795435512, 0.9056059217474234, 0.9267494189391219, 0.9460404248783454, 0.9684216465486508, 0.9700760247364446, 0.976486870859404, 0.9968093749300537, 0.996809374930054, 1.0278367990610184, 1.0342513922128043, 1.0416681263742307, 1.043010406336496, 1.0484093120074813, 1.0583451829721373, 1.0609370984305877, 1.0647007492737106, 1.069892017891733, 1.0698920178917333, 1.0823857981866933, 1.086964916822047, 1.0875266270378605, 1.0954015179394267, 1.1199930738642985, 1.128659596824481, 1.13258861965489, 1.1455655745106867, 1.1466628473548799, 1.1718462342027265, 1.1718462342027267, 1.1848688051864646, 1.1944875409580464, 1.1988236586405632, 1.2032828743012363, 1.2072706930594839, 1.2268752526251396, 1.227270339209744, 1.233187215197361, 1.2432604254555841, 1.2477181113953586, 1.316340565926615, 1.3203577111919331, 1.3301767121341437, 1.3318980762445753, 1.3394926224149204, 1.3444642877522597, 1.3514651917516582, 1.356310419812425, 1.3680443612664352, 1.3680443612664355, 1.3732381022619498, 1.3904475845942559, 1.4158055463262362, 1.4167965575565182, 1.426167060867695, 1.4401379397488283, 1.4449718324642564, 1.4449718324642566, 1.46343809346526, 1.468298972709272, 1.4695742531440867, 1.4857964248564621, 1.5054187818160425, 1.5071935793069187, 1.5071935793069189, 1.511898125633511, 1.5200948935852654, 1.5200948935852658, 1.5264459975587685, 1.5309815431624254, 1.5316230885906628, 1.532266453820614, 1.5398100160422388, 1.5427071576915983, 1.5983083932933284, 1.5996582143881146, 1.6003056780758413, 1.6088606171772617, 1.6165891517339834, 1.6631027886165064, 1.6768991876388994, 1.6901685478294473, 1.6914450635061644, 1.692057405433332, 1.7001507243488743, 1.711459869673841, 1.7116963765528044, 1.731424751231618, 1.7515105724785107, 1.7685706297122712, 1.781217697392433, 1.8003054150740139, 1.8026515152696598, 1.8324479186031153, 1.8439298483141928, 1.855060450666624, 1.8583932238342054, 1.8614490284463874, 1.8614490284463878, 1.8640859651593322, 1.864085965159333, 1.874946528705286, 1.8915138079073068, 1.8945692084228545, 1.9058647605493948, 1.9243048046948723, 2.0276348008218843, 2.0579853246840982, 2.0734652551987334, 2.076046645811853, 2.090041264067292, 2.126498080296336, 2.1531447897213045, 2.15948153164226, 2.1645546451948525, 2.1829034764608397, 2.213171366580022, 2.223179814028547, 2.2862966748661475, 2.28690854817726, 2.2869085481772604, 2.320351517160698, 2.351949762538308, 2.359847555564554, 2.3922609568155395, 2.425989382809414, 2.4413525115197934, 2.4831497106505687, 2.492405964879812, 2.4949081363771275, 2.5329249751914182, 2.5491571323682662, 2.563979654595074, 2.601419345080834, 2.611177323759581, 2.6641838685616284, 2.6666354241823167, 2.680468706309403, 2.6929498323499437, 2.7261310877737017, 2.7534938687972415, 2.753493868797242, 2.754040029757012, 2.8622386492925433, 2.8738268015842574, 2.878157828456251, 2.891128537620075, 2.89439081060529, 2.9265718657678654, 2.928049708893089, 2.939214792527949, 2.9392147925279493, 3.012085018550006, 3.0198274321675904, 3.0761457191596118, 3.0937009187044717, 3.13119793138409, 3.1648116034772116, 3.1813431888911334, 3.2701832155525756, 3.3624182496387927, 3.624141344581362, 3.707704476560127, 4.239430207646306, 4.656938101961846,5]
                        dife = [0.0, 0.020202020202020193, 0.09090909090909091, 0.09270944570168699, 0.10808628325601345, 0.10808628325601347, 0.10827491302293185, 0.10827491302293187, 0.11013905867083648, 0.1111111111111111, 0.11861296312552, 0.12553813998539817, 0.12957670877434, 0.13173876438169718, 0.14696938456699069, 0.1469693845669907, 0.1538618516324144, 0.1665986255670086, 0.17356110390903678, 0.17461067804945057, 0.1746106780494506, 0.17871186921319038, 0.1819798177519557, 0.18294640678379565, 0.18303342472865783, 0.1854723699099141, 0.18571184369578825, 0.18571184369578828, 0.18878775470493478, 0.18892223261787974, 0.1897871133667952, 0.19365130117373594, 0.19592835323158522, 0.2, 0.20000000000000007, 0.20041931352683534, 0.20203050891044214, 0.20327890704543547, 0.20396078054371142, 0.20608041101101565, 0.20827332469084403, 0.20995626366712955, 0.2135415650406262, 0.21354156504062627, 0.2154065922853802, 0.2196917449744467, 0.21969174497444682, 0.2207724979877897, 0.2208004403689453, 0.22213150075601149, 0.2222222222222222, 0.22268088570756164, 0.22879178091082225, 0.2316635341652468, 0.23570226039551584, 0.23636363636363641, 0.2424242424242424, 0.24348706415993265, 0.24454238537017814, 0.2449489742783178, 0.24494897427831783, 0.25155319983192226, 0.256124969497314, 0.2569911368986184, 0.25699113689861847, 0.25798968523169763, 0.265150641496141, 0.26620330112690976, 0.2711401631305156, 0.27114016313051564, 0.2785677655436824, 0.27856776554368246, 0.2800291530012764, 0.28017487250585743, 0.2939387691339814, 0.29416086117105716, 0.29667022624539835, 0.3073330883274691, 0.3091206165165234, 0.30912061651652345, 0.3091206165165235, 0.30912061651652356, 0.3124099870362662, 0.3253681957566187, 0.32796216805215417, 0.3307305792474937, 0.3315805769527909, 0.3333333333333333, 0.34110136816367415, 0.3452298849598449, 0.35136418446315326, 0.3565391723301883, 0.36490485822410695, 0.3671301955694549, 0.3720082303616281, 0.37434294373819704, 0.3758118783397281, 0.376796110173626, 0.38376947813196277, 0.38678159211627433, 0.38873012632301995, 0.38873012632302006, 0.3904272341431856, 0.3922144815614096, 0.392491511725871, 0.39810663004889646, 0.4, 0.4000000000000001, 0.4102005153701265, 0.41182520563948005, 0.4151453709393202, 0.42237424163885756, 0.42344080109194104, 0.4311919881123082, 0.43555162199158576, 0.44050476097818114, 0.45370133122923906, 0.4568491119736483, 0.46168379594441233, 0.4621688003316538, 0.46427960923947065, 0.47140452079103173, 0.4789444175497204, 0.48332183894378294, 0.5183068350973601, 0.5206833117271102, 0.5206833117271104, 0.5285599057070339, 0.5324066022538195, 0.5337583340648402, 0.5447844743169451, 0.5637178175095922, 0.5878775382679627, 0.5965083588871344, 0.6000000000000001, 0.618241233033047, 0.6204014500173847, 0.6256019323641655, 0.6303030303030304, 0.6308400618805604, 0.6446359868604573, 0.6642548967837832, 0.676280542305356, 0.6816972283287582, 0.6852736679604725, 0.6863753427324668, 0.7071067811865476, 0.7348469228349533, 0.7504268757632342, 0.76, 0.7832428663927313, 0.7867733763155768, 0.7924021924300193, 0.7986099033807293, 0.812403840463596, 0.8158431221748457, 0.8330932987633766, 0.8459051693633013, 0.8541662601625049, 0.8666666666666668, 0.8724875746400257, 0.873053390247253, 0.8816775895421333, 0.9130169768410661, 0.9273618495495703, 0.9285592184789413, 0.9579144011862437, 0.966229786334493, 0.9744742172063867, 0.9787538933120364, 0.992191737742481, 1.0, 1.019803902718557, 1.0225458424931373, 1.0366136701947202, 1.0476640682967036, 1.0677078252031311, 1.077032961426901, 1.0808628325601344, 1.0827491302293186, 1.100187812666772, 1.1125133484202439, 1.127435635019184, 1.1335333957072784, 1.1661903789690602, 1.188949115816148, 1.224744871391589, 1.2308948130593151, 1.3224556283251583, 1.55492050529208, 1.5684387141358123, 1.5972198067614582, 1.6377491328717662, 1.6911534525287764, 1.7152378308347012, 1.7204650534085253, 1.7333333333333336, 1.7786109158901102, 1.8037738217415178, 1.8303342472865778, 1.8402299532916357, 1.84413023578342, 1.873392644375439, 1.8867962264113207, 1.9332873557751316, 1.9905331502444823, 2.0452727191912006, 2.0591260281974, 2.103711006768753, 2.23606797749979, 2.449489742783178,8,17]

                        diam = np.array(zz)
                        for (i,j) in enumerate(words):
                            b.append(j)
                            if "Position" in j:
                                b.append("diamet_" +  str(np.argmin(np.abs(np.array(dia) - diam[numin]))))
                                b.append("cent_" +  str(np.argmin(np.abs(np.array(cent) - key_diff[numin]))))
                                b.append("diff_" +  str(np.argmin(np.abs(np.array(dife) - key_dife[numin]))))
                                numin = numin + 1
                        
                        print(b)
                        data = self.data_extractor.words_to_data(b)
                        print(f"Parsed {len(data)} words from midi as efficient REMI.")
                        
                        training_data.append(data)
                    else:
                        data = self.data_extractor.extract_data(path, transposition_step)
                        training_data.append(data)


        return DataSet(training_data, self.dictionary)

    def to_midi(self, data) -> MidiWrapper:
        if self.efficient_remi_config.enabled:
            efficient_words = list(map(lambda x: self.dictionary.data_to_word(x), data))
            words = self.efficient_remi_converter.convert_to_normal_remi(efficient_words)
            data = self.data_extractor.words_to_data(words)

        return MidiToolkitWrapper(self.to_midi_mapper.to_midi(data))
