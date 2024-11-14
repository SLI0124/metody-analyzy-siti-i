import random
import csv
import os
import networkx as nx
from tabulate import tabulate

X_RANGE = (0, 1_000)
Y_RANGE = (0, 1_000)


def generate_graph(n, p):
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


def save_nodes_to_csv(nodes, filename_prefix):
    file_prefix = "../results/task6/"

    if not os.path.exists(os.path.dirname(file_prefix)):
        os.makedirs(os.path.dirname(file_prefix))

    with open(file_prefix + filename_prefix + "_nodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        writer.writerows([[node] for node in nodes])


def save_edges_to_csv(edges, filename_prefix):
    file_prefix = "../results/task6/"

    if not os.path.exists(os.path.dirname(file_prefix)):
        os.makedirs(os.path.dirname(file_prefix))

    with open(file_prefix + filename_prefix + "_edges.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(edges)


def calculate_path_distance(edges):
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


def calculate_diameter_and_radius(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    diameter = 0  # also known as the longest path
    radius = 0  # also known as the shortest path
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        diameter = max(diameter, nx.diameter(subgraph))
        radius = max(radius, nx.radius(subgraph))
    return diameter, radius


def count_weakly_connected_components(edges, n):
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


def calculate_average_clustering_coefficient(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    clustering_coefficients = nx.clustering(G).values()  # calculate clustering coefficient for each node

    if len(clustering_coefficients) > 0:
        average_clustering_coefficient = sum(clustering_coefficients) / len(clustering_coefficients)
    else:
        average_clustering_coefficient = 0.0

    return average_clustering_coefficient


def calculate_graph_properties(edges, n, probability):
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


def save_properties_to_csv(data, filename, headers):
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
    save_nodes_to_csv(list(range(n)), f"graph_n_{n}")

    for p in p_values:
        edges = generate_graph(n, p)
        plot_name = f"Graph with n={n} and p={p}"
        print(plot_name)
        print(f"Number of edges: {len(edges)}")

        if has_graph_multi_edges(edges) or has_graph_loops(edges):
            print("Graph has multi-edges or loops, rerun the generation")
            continue
        else:
            print("Graph does not have multi-edges or loops, saving to CSV")
            save_edges_to_csv(edges, f"graph_n_{n}_p_{p}")
        print()

        data.append(calculate_graph_properties(edges, n, p))

    print(tabulate(data, headers=headers, tablefmt="grid"))
    save_properties_to_csv(data, "graph_properties", headers)


if __name__ == '__main__':
    main()
