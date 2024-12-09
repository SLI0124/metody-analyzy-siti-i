import random
import networkx as nx

from task7 import (create_barabasi_albert_graph,
                   generate_connected_graph,
                   STARTING_NODES_COUNT_RANGE,
                   check_graph_properties)


def calculate_degree_stats(graph, name):
    check_graph_properties(graph, name)
    number_of_connected_components = nx.number_connected_components(graph)

    largest_connected_component = max(nx.connected_components(graph), key=len)
    largest_connected_component_size = len(largest_connected_component)

    average_shortest_path_length = nx.average_shortest_path_length(graph)

    average_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()

    print(f"Number of connected components in the {name} graph: {number_of_connected_components}")
    print(f"Size of the largest connected component in the {name} graph: {largest_connected_component_size}")
    print(f"Average shortest path length in the {name} graph: {average_shortest_path_length}")
    print(f"Average degree in the {name} graph: {average_degree}\n")


def main():
    n = 10_000
    m = 2

    initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
    initial_graph = generate_connected_graph(initial_node_count)
    calculate_degree_stats(initial_graph, "Initial")

    g_ba = create_barabasi_albert_graph(initial_graph, m, n)
    calculate_degree_stats(g_ba, "Barabasi-Albert")


if __name__ == "__main__":
    main()
