import networkx as nx
import matplotlib.pyplot as plt
import random

STARTING_NODES_COUNT_RANGE = (5, 10)


def generate_connected_graph(size):
    """ Generate a connected graph with a given size, not necessarily fully connected """
    if size < 2:
        raise ValueError("The size of the graph must be at least 2 to have some more fun graph")

    G = nx.Graph()
    G.add_nodes_from(range(size))

    # Create a connected graph by connecting all nodes in a random order
    nodes = list(G.nodes)
    random.shuffle(nodes)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])

    # Add additional edges to the graph to make it more populated
    additional_edges = random.randint(size, size + 5)  # number of additional edges
    while G.number_of_edges() < additional_edges:
        u, v = random.sample(list(G.nodes), 2)
        if u != v and not G.has_edge(u, v):  # Ensure no loops and no multi-edges
            G.add_edge(u, v)

    return G


def create_barabasi_albert_graph(m, n):
    pass


def main():
    m_sizes = [2, 3]  # number of edges to attach from a new node to existing nodes
    target_nodes_count = 550
    for m in m_sizes:
        initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
        G = generate_connected_graph(initial_node_count)

        # print the graph
        pos = nx.shell_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.show()


if __name__ == '__main__':
    main()
