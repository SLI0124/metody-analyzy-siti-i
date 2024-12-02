import os
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from task7 import check_graph_properties, generate_connected_graph, create_barabasi_albert_graph, \
    STARTING_NODES_COUNT_RANGE


def print_graph_info(G, name):
    check_graph_properties(G, name)
    print(f"Number of nodes in the {name} graph: {G.number_of_nodes()}")
    print(f"Number of edges in the {name} graph: {G.number_of_edges()}\n")


def random_node_sampling(G, sample_size):
    nodes = random.sample(list(G.nodes()), sample_size)  # randomly select a sample of nodes
    selected_nodes = set(nodes)  # set of selected nodes
    # select edges where both endpoints are in the sample set VS
    edges_in_sample = [(u, v) for u, v in G.edges() if u in selected_nodes and v in selected_nodes]
    subgraph = G.subgraph(selected_nodes).copy()  # create a subgraph from the selected nodes
    subgraph.add_edges_from(edges_in_sample)

    print_graph_info(subgraph, "Random Node Sample")
    return subgraph


def random_edge_sampling(G, sample_size):
    sampled_edges = []
    nodes_in_sample = set()

    while len(nodes_in_sample) < sample_size:
        # Select an edge based on the degree of nodes (degree(u) + degree(v) probability)
        edge = random.choices(list(G.edges()), weights=[G.degree(u) + G.degree(v) for u, v in G.edges()])[0]
        u, v = edge

        # Only add the edge if both nodes are not already in the sample
        if u not in nodes_in_sample or v not in nodes_in_sample:
            sampled_edges.append(edge)
            nodes_in_sample.add(u)
            nodes_in_sample.add(v)

    subgraph = G.subgraph(nodes_in_sample).copy()  # Create the subgraph with nodes in sample
    subgraph.add_edges_from(sampled_edges)  # Add the sampled edges to the subgraph
    print_graph_info(subgraph, "Random Edge Sample")

    return subgraph


def random_walk_sampling(G, start_node, size):
    current_node = start_node
    visited_nodes = {current_node}
    walk = [current_node]

    # Keep walking until we've visited the desired number of nodes
    while len(visited_nodes) < size:
        neighbors = list(G.neighbors(current_node))  # List of neighbors of the current node
        if not neighbors:
            break  # Stop if there are no neighbors (isolated node)
        next_node = random.choice(neighbors)  # Randomly choose a neighbor
        if next_node not in visited_nodes:
            visited_nodes.add(next_node)
            walk.append(next_node)
        current_node = next_node

    subgraph = G.subgraph(visited_nodes).copy()  # Create the subgraph with visited nodes
    print_graph_info(subgraph, "Random Walk Sample")

    return subgraph


def snowball_sampling(G, start_node, size, nv):
    visited_nodes = {start_node}  # Set to track visited nodes
    nodes_to_visit = [start_node]  # List of nodes to visit (starting with the start_node)
    sampled_nodes = [start_node]  # List of sampled nodes

    while len(sampled_nodes) < size and nodes_to_visit:
        current_node = nodes_to_visit.pop(0)  # BFS-like behavior

        neighbors = list(G.neighbors(current_node))  # List of neighbors of the current node

        # Randomly sample `nv` neighbors, but only add them if they haven't been visited
        unvisited_neighbors = [n for n in neighbors if n not in visited_nodes]
        sampled_neighbors = random.sample(unvisited_neighbors, min(nv, len(unvisited_neighbors)))

        # Add sampled neighbors to visited nodes and to nodes to visit
        for neighbor in sampled_neighbors:
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                nodes_to_visit.append(neighbor)
                sampled_nodes.append(neighbor)

        if len(sampled_nodes) >= size:  # Stop if we've sampled enough nodes
            break

    # if the number of sampled nodes is more than the desired size, remove the extra nodes
    if len(sampled_nodes) > size:
        sampled_nodes = random.sample(sampled_nodes, size)

    # Create a subgraph of the sampled nodes
    subgraph = G.subgraph(sampled_nodes).copy()
    print_graph_info(subgraph, "Snowball Sample")

    return subgraph


def calculate_ks_statistic(original_graph, sampled_graph):
    original_degrees = [deg for node, deg in original_graph.degree()]  # degree distribution of original graph
    sampled_degrees = [deg for node, deg in sampled_graph.degree()]  # degree distribution of sampled graph
    ks_statistic, p_value = ks_2samp(original_degrees, sampled_degrees)  # calculate KS statistic and p-value

    return ks_statistic, p_value


def plot_degree_distributions(graph_dict, save_directory):
    plt.figure(figsize=(12, 8))
    for graph_name, G_sample in graph_dict.items():
        degrees = [deg for node, deg in G_sample.degree()]
        degree_counts = {deg: degrees.count(deg) for deg in set(degrees)}
        degree_values = sorted(degree_counts.keys())
        counts = [degree_counts[deg] for deg in degree_values]
        plt.loglog(degree_values, counts, label=graph_name, marker='o', linestyle='-')

    plt.title("Degree Distributions (Log-Log Scale)")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.grid(True)
    plt.xlim(left=1)

    os.makedirs(save_directory, exist_ok=True)
    plt.savefig(f"{save_directory}/degree_distributions.png")
    plt.show()


def plot_ks_statistics(ks_results, graph_dict, save_directory):
    for (graph_name, G_sample), (ks_statistic, p_value) in zip(graph_dict.items(), ks_results):
        if graph_name == "Barabasi-Albert":
            continue  # Skip the original Barabasi-Albert model

        plt.figure(figsize=(12, 8))

        original_degrees = [deg for node, deg in graph_dict["Barabasi-Albert"].degree()]
        sampled_degrees = [deg for node, deg in G_sample.degree()]

        # Calculate the empirical CDFs
        original_cdf = np.cumsum(np.bincount(original_degrees)) / len(original_degrees)
        sampled_cdf = np.cumsum(np.bincount(sampled_degrees)) / len(sampled_degrees)

        # Plot the CDFs
        plt.step(range(len(original_cdf)), original_cdf, label="Original", where='post')
        plt.step(range(len(sampled_cdf)), sampled_cdf, label="Sampled", linestyle='--', where='post')

        plt.title(f"Empirical CDFs of Degree Distributions for {graph_name} (KS: {ks_statistic:.4f})")
        plt.xlabel("Degree")
        plt.ylabel("Cumulative Probability")
        plt.legend()
        plt.grid(True)

        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(f"{save_directory}/ks_statistics_curve_{graph_name}.png")
        plt.show()


def main():
    n = 5_000  # Number of nodes in the graph
    m = 2  # Number of edges to attach from a new node to existing nodes
    desired_size = int(n * 0.15)

    initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
    G = generate_connected_graph(initial_node_count)
    check_graph_properties(G, "initial")
    print(f"Number of nodes in the graph: {G.number_of_nodes()}")
    print(f"Number of edges in the graph: {G.number_of_edges()}\n")

    G_BA = create_barabasi_albert_graph(G, m, n)
    print(f"Generated Barabasi-Albert graph with m={m}, n={n}")
    print(f"Number of nodes in the graph: {G_BA.number_of_nodes()}")
    print(f"Number of edges in the graph: {G_BA.number_of_edges()}")

    average_degree = sum(dict(G_BA.degree()).values()) / G_BA.number_of_nodes()
    print(f"Average degree of the Barabasi-Albert graph: {average_degree}\n")

    start_node = random.choice(list(G_BA.nodes()))

    graph_dict = {
        "Barabasi-Albert": G_BA,
        "Random Node Sample": random_node_sampling(G_BA, desired_size),
        "Random Edge Sample": random_edge_sampling(G_BA, desired_size),
        "Random Walk Sample": random_walk_sampling(G_BA, start_node, desired_size),
        "Snowball Sample": snowball_sampling(G_BA, start_node, desired_size, 5)
    }

    ks_results = []
    for graph_name, G_sample in graph_dict.items():
        ks_statistic, p_value = calculate_ks_statistic(G_BA, G_sample)
        ks_results.append((ks_statistic, p_value))
        print(f"KS Statistic for {graph_name}: {ks_statistic}, p-value: {p_value}")

    save_directory = "../results/task8"
    plot_degree_distributions(graph_dict, save_directory)
    plot_ks_statistics(ks_results, graph_dict, save_directory)


if __name__ == "__main__":
    main()
