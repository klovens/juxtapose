from netwalk.translator import IDCovertor
import os
import argparse
import json
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate datasets')
    parser.add_argument('-c', '--config', metavar='JSON file path',
                        action='store', required=True,
                        help='Path to a config file')
    args = parser.parse_args()
    # read config file
    with open(args.config) as fin:
        params = json.load(fin)

    edge_list_file_addresses = glob.glob(os.path.join(params['experiment_name'],
                                             'anchored_*.csv'))
    print(os.listdir(params['experiment_name']))
    #output_file_address = 'Line/translated_line.csv'
    convertor = IDCovertor(edge_list_file_addresses, sep=',')
    for edge_list_file_address in edge_list_file_addresses:
        dir_path, file_name = os.path.split(edge_list_file_address)
        output_file_address = os.path.join(dir_path, f'translated_{file_name}')
        convertor.translate(edge_list_file_address, output_file_address, sep=',')

    convertor_file = os.path.join(params['experiment_name'],
                                             'IDConvertor.json')
    convertor.save(convertor_file)
    con = IDCovertor.load(convertor_file)
    assert con.id2int == convertor.id2int
    assert con.int2id == convertor.int2id
    assert con.ids == convertor.ids

