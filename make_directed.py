import networkx as nx

G = nx.read_edgelist("/home/farhad/Network/juxt/brain_heart_data/heart_1.csv", delimiter=',',nodetype=str, data=(('cor',float),))
#print(G.edges(data=True))
T = nx.algorithms.tree.mst.minimum_spanning_tree(G, weight='cor')
edj = T.edges()
print(edj)
address = 'test/data/heart_directed_1.txt'
with open(address, 'w') as fout:
    for e in edj:
        # write edges to file
        node_1 = e[0]
        node_2 = e[1]
        fout.write('{}\t{}\n'.format(str(node_1),str(node_2)))
