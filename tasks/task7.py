import networkx as nx
import random
from tabulate import tabulate
import matplotlib.pyplot as plt

from task6 import save_edges_to_csv, save_nodes_to_csv, save_properties_to_csv, calculate_path_distance, \
    calculate_diameter_and_radius, count_weakly_connected_components, save_plot, \
    calculate_average_clustering_coefficient, has_graph_loops, has_graph_multi_edges

STARTING_NODES_COUNT_RANGE = (5, 10)


def generate_connected_graph(size: int) -> nx.Graph:
    """ Initialize connected graph with a given size, not necessarily fully connected """
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


def create_barabasi_albert_graph(G: nx.Graph, m: int, n: int) -> nx.Graph:
    """
    Create a Barabasi-Albert graph with m edges to attach from a new node to existing nodes,
    with probability proportional to the degree of existing nodes.
    :param G: networkx graph
    :param m: number of edges to attach from a new node to existing nodes
    :param n: number of nodes in the graph
    :return: networkx graph
    """
    while G.number_of_nodes() < n:
        new_node = G.number_of_nodes()
        G.add_node(new_node)

        # Compute cumulative probabilities for degree-proportional selection
        degrees = dict(G.degree())
        total_degree = sum(degrees.values())
        cumulative_probs = []
        cumulative_sum = 0.0

        for node, degree in degrees.items():
            cumulative_sum += degree / total_degree
            cumulative_probs.append((node, cumulative_sum))

        # Select m unique targets
        targets = set()
        while len(targets) < m:
            rand_val = random.random()
            for node, cum_prob in cumulative_probs:
                if rand_val <= cum_prob:
                    if node != new_node:  # Avoid self-loops
                        targets.add(node)
                    break

        # Connect new node to targets
        for target in targets:
            G.add_edge(new_node, target)

    return G


def calculate_graph_properties(G: nx.Graph) -> dict:
    pass


def main():
    m_sizes = [2, 3]
    target_nodes_count = 550
    headers = [
        "Number of nodes", "Number of edges", "Average degree",
        "Average distance", "Average clustering coefficient", "The longest path",
        "The shortest path", "Number of connected components", "Largest component size",
        "Average component size"
    ]
    directory_prefix = "../results/task7/"
    data = []
    save_nodes_to_csv(list(range(target_nodes_count)), "nodes", directory_prefix)
    for m in m_sizes:
        initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
        G = generate_connected_graph(initial_node_count)
        BA_G = create_barabasi_albert_graph(G, m, target_nodes_count)
        print(f"Generated Barabasi-Albert graph with m={m}, n={target_nodes_count}")
        print(f"Number of nodes: {BA_G.number_of_nodes()}")
        if has_graph_loops(BA_G) or has_graph_multi_edges(BA_G):
            print("Graph has loops or multi-edges")
            continue
        else:
            print("Graph has loops or multi-edges")
            save_edges_to_csv(list(BA_G.edges), f"BA_{m}", directory_prefix)
        # data.append(calculate_graph_properties(BA_G))

    # print(tabulate(data, headers=headers, tablefmt="grid"))
    # save_properties_to_csv(data, "graph_properties", headers, directory_prefix)


if __name__ == '__main__':
    main()
