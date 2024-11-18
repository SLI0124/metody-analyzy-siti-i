import random
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate


def generate_graph(n: int, p: float) -> list:
    """Generate a random graph with n nodes and probability p
    :param n: number of nodes
    :param p: probability of edge creation
    :return: list of edges
    """
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() <= p:
                # Check if the edge adds loops or multi-edges
                # i != j  ensures no loops
                # (i, j) not in edges and (j, i) not in edges ensures no multi-edges
                if i != j and (i, j) not in edges and (j, i) not in edges:
                    edges.add((i, j))
    return list(edges)


def has_graph_multi_edges(edges):
    return len(edges) != len(set(edges))


def has_graph_loops(edges):
    return any([source == target for source, target in edges])


def save_nodes_to_csv(nodes: list, filename_prefix: str, directory_prefix: str) -> None:
    """
    Save the nodes to a CSV file with their IDs.
    :param nodes: list of nodes
    :param filename_prefix: prefix for the filename
    :param directory_prefix: prefix for the directory
    :return: None
    """
    if not os.path.exists(os.path.dirname(directory_prefix)):
        os.makedirs(os.path.dirname(directory_prefix))

    with open(directory_prefix + filename_prefix + "_nodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        writer.writerows([[node] for node in nodes])


def save_edges_to_csv(edges: list, filename_prefix: str, directory_prefix: str) -> None:
    """
    Save the edges to a CSV file, also save the nodes with their degrees, closeness centrality,
    and clustering coefficient. Function create networkx graph from edges and calculate the attributes.
    :param edges: list of edges
    :param filename_prefix: prefix for the filename
    :param directory_prefix: prefix for the directory
    :return: None
    """
    if not os.path.exists(os.path.dirname(directory_prefix)):
        os.makedirs(os.path.dirname(directory_prefix))

    with open(directory_prefix + filename_prefix + "_edges.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(edges)

    # save nodes with their degrees, closeness centrality, and clustering coefficient
    G = nx.Graph()
    G.add_edges_from(edges)

    clustering_coefficient = nx.clustering(G)
    closeness_centrality = nx.closeness_centrality(G)
    degrees = dict(G.degree())

    # sort them by id
    nodes = sorted(G.nodes())

    with open(directory_prefix + filename_prefix + "_attributes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Degree", "Closeness centrality", "Clustering coefficient"])
        for node in nodes:
            writer.writerow([node, degrees[node], closeness_centrality[node], clustering_coefficient[node]])


def calculate_path_distance(edges: list) -> float:
    """
    Calculate the average shortest path length of the graph. Function creates a networkx graph from edges.
    If the graph is connected, it calculates the average shortest path length. Otherwise, it calculates the average
    :param edges: list of edges
    :return: average shortest path length
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:  # Calculate the average shortest path length for each connected component and average them
        total_distance = 0
        num_pairs = 0
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            total_distance += nx.average_shortest_path_length(subgraph) * len(component) * (len(component) - 1) / 2
            num_pairs += len(component) * (len(component) - 1) / 2
        return total_distance / num_pairs if num_pairs > 0 else float('inf')


def calculate_diameter_and_radius(edges: list) -> tuple:
    """
    Calculate the diameter and radius of the graph. Function creates a networkx graph from edges.
    :param edges: list of edges
    :return: tuple of diameter and radius
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    diameter = 0  # also known as the longest path
    radius = 0  # also known as the shortest path
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        diameter = max(diameter, nx.diameter(subgraph))
        radius = max(radius, nx.radius(subgraph))
    return diameter, radius


def count_weakly_connected_components(edges: list, n: int) -> tuple:
    """
    Count the number of connected components, the size of the largest connected component, and the average size of the
    connected components. Function creates a networkx graph from edges.
    :param edges: list of edges
    :param n: number of nodes
    :return: tuple of number of connected components, size of the largest connected component, and average size of the
    connected components
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    components = list(nx.connected_components(G))  # Find weakly connected components

    # Include isolated nodes, meaning nodes that are not connected to any other node
    all_nodes = set(range(n))
    isolated_nodes = all_nodes - set(G.nodes())
    for isolated_node in isolated_nodes:
        components.append({isolated_node})

    component_count = len(components)  # Number of connected components
    component_sizes = [len(component) for component in components]  # Sizes of connected components
    largest_component = max(component_sizes) if component_sizes else 0  # Size of the largest connected component

    return component_count, largest_component, sum(component_sizes) / len(component_sizes)


def save_plot(fig: plt.Figure, directory_prefix: str, filename: str) -> None:
    """Save the plot to the specified directory.
    :param fig: matplotlib figure
    :param directory_prefix: prefix for the directory
    :param filename: filename
    :return: None
    """
    save_path = os.path.join(directory_prefix, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.clf()


def plot_degree_distribution(edges: list, probability: float, directory_prefix: str) -> None:
    """Plot the degree distribution.
    :param edges: list of edges
    :param probability: probability of edge creation
    :param directory_prefix: prefix for the directory
    :return: None
    """
    degrees_distribution = nx.degree_histogram(nx.Graph(edges))
    plt.bar(range(len(degrees_distribution)), degrees_distribution, width=1, edgecolor='black', color='C0')
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title(f"Degree distribution for p={probability}")
    save_plot(plt.gcf(), directory_prefix, f"Degree_distribution_p_{probability}.png")


def plot_components_distribution(edges, probability, directory_prefix):
    """Plot the components' distribution. Function creates a networkx graph from edges.
    :param edges: list of edges
    :param probability: probability of edge creation
    :param directory_prefix: prefix for the directory
    :return: None
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    components = nx.connected_components(G)
    components_sizes = [len(component) for component in components]
    plt.hist(components_sizes, bins=10, edgecolor='black', color='C0')
    plt.xlabel("Component size")
    plt.ylabel("Number of components")
    plt.title(f"Components distribution for p={probability}")
    save_plot(plt.gcf(), directory_prefix, f"Components_distribution_p_{probability}.png")


def calculate_average_clustering_coefficient(edges: list) -> float:
    """
    Calculate the average clustering coefficient of the graph. Function creates a networkx graph from edges.
    :param edges: list of edges
    :return: average clustering coefficient
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    clustering_coefficients = nx.clustering(G).values()  # calculate clustering coefficient for each node

    if len(clustering_coefficients) > 0:
        average_clustering_coefficient = sum(clustering_coefficients) / len(clustering_coefficients)
    else:
        average_clustering_coefficient = 0.0

    return average_clustering_coefficient


def calculate_graph_properties(edges: list, n: int, probability: float) -> list:
    """
    Calculate the properties of the graph. Function creates a networkx graph from edges.
    :param edges: list of edges
    :param n: number of nodes
    :param probability: probability of edge creation
    :return: list of graph properties: number of nodes, probability, number of edges, average degree, average distance,
    average clustering coefficient, the longest path, the shortest path, number of connected components, size of the
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    number_of_nodes = n
    number_of_edges = len(edges)
    average_degree = sum(dict(G.degree()).values()) / number_of_nodes
    average_path_distance = calculate_path_distance(edges)
    clustering_coefficient = calculate_average_clustering_coefficient(edges)
    longest_path, shortest_path = calculate_diameter_and_radius(edges)

    num_connected_components, largest_component_size, average_component_size = (
        count_weakly_connected_components(edges, n))

    return [
        number_of_nodes, probability, number_of_edges, average_degree,
        average_path_distance, clustering_coefficient, longest_path,
        shortest_path, num_connected_components, largest_component_size,
        average_component_size
    ]


def save_properties_to_csv(data: list, filename: str, headers: list) -> None:
    """
    Save the properties of the graph to a CSV file.
    :param data: list of graph properties
    :param filename: filename
    :param headers: list of headers
    :return: None
    """
    file_prefix = "../results/task6/"
    os.makedirs(file_prefix, exist_ok=True)

    with open(file_prefix + filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def main():
    n = 550
    p_values = [0.001, 0.0059, 0.01]
    data = []

    headers = [
        "Number of nodes", "Probability", "Number of edges", "Average degree",
        "Average distance", "Average clustering coefficient", "The longest path",
        "The shortest path", "Number of connected components", "Largest component size",
        "Average component size"
    ]

    # save the nodes to a CSV file once, we don't need to do it for each graph
    save_nodes_to_csv(list(range(n)), f"graph_n_{n}", "../results/task6/")

    for p in p_values:
        edges = generate_graph(n, p)
        print(f"Graph with n={n} and p={p}")
        print(f"Number of edges: {len(edges)}")

        if has_graph_multi_edges(edges) or has_graph_loops(edges):
            print("Graph has multi-edges or loops, rerun the generation")
            continue
        else:
            print("Graph does not have multi-edges or loops, saving to CSV")
            save_edges_to_csv(edges, f"graph_n_{n}_p_{p}", "../results/task6/")
        print()

        data.append(calculate_graph_properties(edges, n, p))
        plot_degree_distribution(edges, p, "../results/task6/")
        plot_components_distribution(edges, p, "../results/task6/")

    print(tabulate(data, headers=headers, tablefmt="grid"))
    save_properties_to_csv(data, "graph_properties", headers)


if __name__ == '__main__':
    main()
