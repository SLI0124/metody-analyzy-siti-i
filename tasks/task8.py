import random

from task7 import check_graph_properties, generate_connected_graph, create_barabasi_albert_graph, \
    STARTING_NODES_COUNT_RANGE


def main():
    n = 5_000  # Number of nodes in the graph
    m = 2  # Number of edges to attach from a new node to existing nodes
    # directory_prefix = "../results/task8/"

    # Generate an initial connected graph
    initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
    G = generate_connected_graph(initial_node_count)
    check_graph_properties(G, "initial")
    print(f"Number of nodes in the graph: {G.number_of_nodes()}")
    print(f"Number of edges in the graph: {G.number_of_edges()}\n")

    # Create a Barabási–Albert graph
    G_BA = create_barabasi_albert_graph(G, m, n)
    print(f"Generated Barabasi-Albert graph with m={m}, n={n}")
    print(f"Number of nodes in the graph: {G_BA.number_of_nodes()}")
    print(f"Number of edges in the graph: {G_BA.number_of_edges()}")

    # Calculate the average degree
    average_degree = sum(dict(G_BA.degree()).values()) / G_BA.number_of_nodes()
    print(f"Average degree of the Barabasi-Albert graph: {average_degree}\n")


if __name__ == "__main__":
    main()
