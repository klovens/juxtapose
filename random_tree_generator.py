import random
import networkx as nx

START_SIZE=50
CURRENT_SIZE=50
FINAL_SIZE=200
NODE_STEP=10

while CURRENT_SIZE <= FINAL_SIZE:
    #print(CURRENT_SIZE - 1)
    address = 'test/data/random_tree_{}.csv'.format(CURRENT_SIZE)
    with open(address, 'w') as fout:
        if CURRENT_SIZE == START_SIZE:
            G = nx.generators.trees.random_tree(START_SIZE)
            while nx.number_connected_components(G) > 1:
                nx.generators.trees.random_tree(START_SIZE)
            edj = list(G.edges())
            n = list(G.nodes())
            for e in edj:
                # write edges to file
                fout.write('{},{},1\n'.format(str(e[0]),str(e[1])))
                fout.write('{},{},1\n'.format(str(e[1]),str(e[0])))
        elif CURRENT_SIZE != START_SIZE:
            # read in the previous graph and write it to file
            previous_address = 'test/data/random_tree_{}.csv'.format(CURRENT_SIZE-NODE_STEP)
            file_previous = open(previous_address, 'r')
            Lines = file_previous.readlines()
            fout.writelines(Lines)

            G = nx.generators.trees.random_tree(NODE_STEP)
            while nx.number_connected_components(G) > 1:
                nx.generators.trees.random_tree(NODE_STEP)
            edj = list(G.edges())

            for e in edj:
                # write edges to file
                node_1 = e[0] + CURRENT_SIZE-NODE_STEP - 1
                node_2 = e[1] + CURRENT_SIZE- NODE_STEP - 1
                fout.write('{},{},1\n'.format(str(node_1),str(node_2)))
                fout.write('{},{},1\n'.format(str(node_2),str(node_1)))
            # connect a node in graph to a random node in the original graph
            rand_node = random.randint(0, CURRENT_SIZE-1-NODE_STEP)
            #print(rand_node)
            print(CURRENT_SIZE,rand_node,node_1)
            fout.write('{},{},1\n'.format(str(rand_node), str(node_1)))
            fout.write('{},{},1\n'.format(str(node_1), str(rand_node)))
    CURRENT_SIZE= CURRENT_SIZE + NODE_STEP


address = 'test/data/test_PPI.txt'
with open(address, 'w') as fout:
    G = nx.scale_free_graph(100)
    edj = list(G.edges())

    for e in edj:
        # write edges to file
        node_1 = e[0]
        node_2 = e[1]
        fout.write('{}\t{}\t1\n'.format(str(node_1),str(node_2)))
