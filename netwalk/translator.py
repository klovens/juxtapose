import json
import os
import numpy as np
import pandas as pd



class ID2NameTranslator(object):
    def __init__(self, vocab_file_address, sep=','):
        assert os.path.isfile(vocab_file_address)
        df = pd.read_csv(vocab_file_address, sep=sep)
        df.columns = ['ID', 'Name']
        self.df = df
        self.ids = df.iloc[:, 0].values
        self.names = df.iloc[:, 1].values
        self.__id2name = {ID: name for ID, name in zip(self.ids, self.names)}

    def id2name(self, ensemble_id, default=''):
        return self.__id2name.get(ensemble_id, default)

    def names2ids(self, names):
        selected = self.df[self.df.iloc[:, 1].isin(names)]
        data = []
        for name in names:
            data.extend(selected[selected.iloc[:,1] == name].values)
        data = pd.DataFrame.from_records(data, columns=['ID', 'Name'])
        return list(data.ID), list(data.Name)


class IDCovertor(object):
    def __init__(self, edge_list_file_addresses, sep=','):
        idset = set()
        for edge_list_file_address in edge_list_file_addresses:
            with open(edge_list_file_address) as fin:
                for line in fin:
                    id1, id2, _ = line.strip().split(sep)
                    idset.add(id1)
                    idset.add(id2)
        n = len(idset)
        self.ids = sorted(idset)
        self.id2int = {a_id: i for a_id, i in zip(self.ids, range(n))}
        self.int2id = {i: a_id for a_id, i in zip(self.ids, range(n))}

    def ids2ints(self, ids):
        return [self.id2int[x] for x in ids if x in self.ids]

    def ints2ids(self, ints):
        return [self.int2id[int(x)] for x in ints]

    def save(self, json_file_address):
        with open(json_file_address, 'w') as fout:
            json.dump(self.id2int, fout)

    @classmethod
    def load(cls, json_file_address):
        with open(json_file_address) as fin:
            id2int = json.load(fin)
        convertor = IDCovertor([])
        convertor.id2int = id2int
        convertor.int2id = {i: a_id for a_id, i in id2int.items()}
        convertor.ids = sorted(id2int.keys())
        return convertor

    def translate(self, input_file_address, output_file_address, sep=','):
        df = pd.read_csv(input_file_address, sep=sep, header=None)
        df.columns = ['ID1', 'ID2', 'Cor']
        with open(output_file_address, 'w') as fout:
            for id1, id2, cor in df.itertuples(index=False):
                fout.write('{}{}{}{}{}\n'.format(self.id2int[id1], sep,
                                                 self.id2int[id2], sep,
                                                 cor))


def vocab2id_and_name(vocab, id_convertor_file_path, id2name_translator_file, default_name='', sep=','):
    id2name_translator = ID2NameTranslator(id2name_translator_file, sep=sep)
    id_convertor = IDCovertor.load(id_convertor_file_path)
    id_names = {}
    for k in vocab:
        ensemble_id = id_convertor.int2id[int(k)]
        name = id2name_translator.id2name(ensemble_id, default_name)
        id_names[k] = (ensemble_id, name)
    return id_names


if __name__ == '__main__':
    ensemble_id_name_file = '../skeletal_data/mouse.vocab'
   # trans = ID2NameTranslator(ensemble_id_name_file, sep=',')
   # assert trans.id2name('ENSMUSG00000064372') == 'mt-Tp'
   # assert trans.id2name('ENSMUSG00000106796') == 'AC124394.4'
   # edge_list_file_addresses = ['../Skeletal_Cells/anchored_chicken_imm_0.csv',
   #                             '../Skeletal_Cells/anchored_chicken_ost_0.csv',
   #                             '../Skeletal_Cells/anchored_gar_imm_0.csv',
   #                             '../Skeletal_Cells/anchored_gar_ost_0.csv',
   #                             '../Skeletal_Cells/anchored_frog_imm_0.csv',
   #                             '../Skeletal_Cells/anchored_frog_ost_0.csv',
   #                             '../Skeletal_Cells/anchored_mouse_imm_0.csv',
   #                             '../Skeletal_Cells/anchored_mouse_ost_0.csv']
   # output_file_address = '../skeletal_data/translated_chicken_imm.csv'
   # convertor = IDCovertor(edge_list_file_addresses, sep=',')
   # for edge_list_file_address in edge_list_file_addresses:
   #     dir_path, file_name = os.path.split(edge_list_file_address)
   #     output_file_address = os.path.join(dir_path, f'translated_{file_name}')
   #     convertor.translate(edge_list_file_address, output_file_address, sep=',')
    convertor_file = '../skeletal_data/IDConvertor.json'
   # convertor.save(convertor_file)
   # con = IDCovertor.load(convertor_file)
   # assert con.id2int == convertor.id2int
   # assert con.int2id == convertor.int2id
   # assert con.ids == convertor.ids
   # # Test vocab2id_and_name
    v2id_name = vocab2id_and_name(['19210', '19211'], convertor_file, ensemble_id_name_file, default_name='', sep=',')
    assert v2id_name['19210'][0] == 'ENSMUSG00000114019'
    assert v2id_name['19211'][0] == 'ENSMUSG00000114025'
    print(v2id_name)
