import pandas as pd
import multiprocessing as mp


def load(address):
    genes = set()
    with open(address) as  fin:
        for line in fin:
            line = line.strip()
            if line == '':
                continue
            gene_a, gene_b, _ =  line.split(',')
            genes.add(gene_a)
            genes.add(gene_b)
    return genes

def get_common_genes(edge_lists, genes_of_interest):
    pool = mp.Pool(min(mp.cpu_count(), len(edge_llists)))
    profiles_genes = [pool.apply(load, args=(address, )) for address in edge_lists]
    pool.close()
    common_genes = set(genes_of_interest)
    for genes in profiles_genes:
        common_genes =  genes & common_genes
    return common_genes


if __name__ == '__main__':
    output_address = 'data/common_cortex_genes.csv'
    edge_llists = [
        'pcortex_data/network_1_12_chimpanzee.csv',
        'pcortex_data/network_1_12_human.csv',
        'pcortex_data/network_1_12_macaque.csv',
        'pcortex_data/network_1_12_mouse.csv']
    #genes_of_interest = '/home/fam918/Documents/CodeRepos/WALKS/netwalk/data/homeostasis_genes.csv'
    #output_address = 'heart_brain_shared.csv'
    #edge_llists = ['/home/farhad/Network/netwalk/data/network_1_200_heart.csv',
    #                '/home/farhad/Network/netwalk/data/network_2_200_heart.csv',
    #                '/home/farhad/Network/netwalk/data/network_3_200_heart.csv',
    #                '/home/farhad/Network/netwalk/data/network_1_200_brain.csv',
    #                '/home/farhad/Network/netwalk/data/network_2_200_brain.csv',
    #                '/home/farhad/Network/netwalk/data/network_3_200_brain.csv']
    genes_of_interest = 'data/cellular_homeostasis.csv'
    with open(genes_of_interest) as f:
        lines = f.read().splitlines()
    common_genes = get_common_genes(edge_llists, lines)
    with open(output_address, 'w') as fout:
        for gene in common_genes:
            fout.write('{}\n'.format(gene))
