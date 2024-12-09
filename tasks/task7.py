import networkx as nx
import random
from tabulate import tabulate
import matplotlib.pyplot as plt

from task6 import save_edges_to_csv, save_nodes_to_csv, save_properties_to_csv

STARTING_NODES_COUNT_RANGE = (5, 10)


def has_graph_loops(G: nx.Graph) -> bool:
    """Checks if there are any loops in the graph."""
    seen_edges = set()
    for u, v in G.edges:
        if u == v:
            return True
        edge = tuple(sorted((u, v)))  # Ensure the order is consistent
        if edge in seen_edges:
            return True
        seen_edges.add(edge)
    return False


def has_graph_multi_edges(G: nx.Graph) -> bool:
    """Checks if there are any multi-edges in the graph."""
    seen_edges = set()
    for u, v in G.edges:
        edge = tuple(sorted((u, v)))  # Ensure the order is consistent
        if edge in seen_edges:
            return True
        seen_edges.add(edge)
    return False


def generate_connected_graph(size: int) -> nx.Graph:
    """Initialize a connected graph with a given size, avoiding loops and multi-edges."""
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
                    # if node != new_node:  # Avoid self-loops and multi-edges
                    if node != new_node and not G.has_edge(new_node, node):  # Avoid self-loops and multi-edges
                        targets.add(node)
                    break

        # Connect new node to targets
        for target in targets:
            G.add_edge(new_node, target)

    return G


def calculate_graph_properties(G: nx.Graph, initial_node_count: int, initial_edge_count: int) -> list:
    """
    Calculate graph properties for a given graph: initial number of nodes, initial number of edges, number of nodes,
    number of edges, average degree, average distance, average clustering coefficient, longest path, shortest path,
    number of connected components, largest component size, average component size.
    :param G: networkx graph
    :param initial_node_count: initial number of nodes in the graph before Barabasi-Albert model
    :param initial_edge_count: initial number of edges in the graph before Barabasi-Albert model
    :return: list of graph properties
    """
    number_of_nodes = G.number_of_nodes()
    number_of_edges = G.number_of_edges()
    average_degree = sum(dict(G.degree()).values()) / number_of_nodes
    average_distance = nx.average_shortest_path_length(G)
    average_clustering_coefficient = nx.average_clustering(G)
    longest_path = nx.diameter(G)
    shortest_path = nx.radius(G)
    number_of_connected_components = nx.number_connected_components(G)
    largest_component_size = max(len(c) for c in nx.connected_components(G))
    average_component_size = number_of_nodes / number_of_connected_components

    return [
        initial_node_count, initial_edge_count, number_of_nodes, number_of_edges, average_degree, average_distance,
        average_clustering_coefficient, longest_path, shortest_path, number_of_connected_components,
        largest_component_size, average_component_size
    ]


def plot_degree_distribution(G: nx.Graph, m: int, file_name: str, directory_prefix: str):
    """Plot the bar plot as histogram"""
    degrees_distribution = dict(G.degree())

    # Apply a more visually appealing style
    plt.figure(figsize=(10, 6))  # Set a larger figure size

    # Create a bar plot for the degree distribution
    plt.bar(degrees_distribution.keys(), degrees_distribution.values(), color='C0', width=1.0)

    # Enhance plot labels and title
    plt.xlabel("Node Degree", fontsize=14)
    plt.ylabel("Number of Nodes", fontsize=14)
    plt.title(f"Degree Distribution of Barabási-Albert Graph (m={m})", fontsize=16)

    # Add grid lines for better readability
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Save the plot to the specified file
    plt.savefig(f"{directory_prefix}{file_name}.png")
    plt.close()


def check_graph_properties(G: nx.Graph, graph_name: str):
    if has_graph_loops(G) and has_graph_multi_edges(G):
        raise ValueError(f"{graph_name} graph has loops and multi-edges")
    elif has_graph_loops(G):
        raise ValueError(f"{graph_name} graph has loops")
    elif has_graph_multi_edges(G):
        raise ValueError(f"{graph_name} graph has multi-edges")
    else:
        print(f"{graph_name} graph has no loops or multi-edges\n")


def main():
    m_sizes = [2, 3]
    target_nodes_count = 550
    headers = [
        "Initial number of nodes", "Initial number of edges", "Number of nodes", "Number of edges", "Average degree",
        "Average distance", "Average clustering coefficient", "Longest path", "Shortest path",
        "Number of connected components", "Largest component size", "Average component size"
    ]
    directory_prefix = "../results/task7/"
    data = []
    save_nodes_to_csv(list(range(target_nodes_count)), "nodes", directory_prefix)
    for m in m_sizes:
        initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
        G = generate_connected_graph(initial_node_count)
        check_graph_properties(G, "Initial graph")
        initial_edge_count = G.number_of_edges()
        BA_G = create_barabasi_albert_graph(G, m, target_nodes_count)
        print(f"Generated Barabasi-Albert graph with m={m}, n={target_nodes_count}")
        print(f"Number of nodes: {BA_G.number_of_nodes()}")
        print(f"Number of edges: {BA_G.number_of_edges()}")
        print(f"Number of initial nodes: {initial_node_count}")
        print(f"Number of initial edges: {initial_edge_count}")
        # if graph has loops or multi-edges, raise an error, otherwise save the nodes and edges to CSV
        check_graph_properties(BA_G, "Barabasi-Albert graph")
        save_edges_to_csv(list(BA_G.edges), f"BA_{m}", directory_prefix)
        graph_properties = calculate_graph_properties(BA_G, initial_node_count, initial_edge_count)
        data.append(graph_properties)
        plot_degree_distribution(BA_G, m, f"BA_{m}_degree_distribution", directory_prefix)

    print(tabulate(data, headers=headers, tablefmt="grid"))
    save_properties_to_csv(data, "graph_properties", headers, directory_prefix)


if __name__ == '__main__':
    main()
