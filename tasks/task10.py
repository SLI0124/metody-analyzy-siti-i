import os
import networkx as nx
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from task6 import generate_random_graph
from task7 import (create_barabasi_albert_graph,
                   generate_connected_graph,
                   STARTING_NODES_COUNT_RANGE)


def remove_nodes(graph, strategy, graph_name="Unknown"):
    """Remove nodes from the graph based on the given strategy and return metrics."""
    nodes = list(graph.nodes())
    removed_count = 0
    total_nodes = len(nodes)
    results = []

    # Initial calculation of the largest connected component and average shortest path length
    largest_cc = max(nx.connected_components(graph), key=len) if graph.number_of_nodes() > 0 else []
    largest_cc_size = len(largest_cc) / total_nodes
    avg_shortest_path = calculate_average_shortest_path(graph)
    results.append((removed_count, largest_cc_size, avg_shortest_path))

    while graph.number_of_nodes() > 0:
        node = select_node(graph, strategy, nodes)
        graph.remove_node(node)
        nodes.remove(node)
        removed_count += 1

        # Update every 10 nodes or when graph is empty
        if removed_count % 10 == 0 or graph.number_of_nodes() == 0:
            print(f"{strategy.capitalize()} removal ({graph_name}): {removed_count}/{total_nodes} nodes removed.")

        # Recalculate after every 10 nodes removed
        if removed_count % 10 == 0:
            if graph.number_of_nodes() > 0:
                # Find the largest connected component (considering all components)
                largest_cc = max(nx.connected_components(graph), key=len)
                largest_cc_size = len(largest_cc) / total_nodes
                avg_shortest_path = calculate_average_shortest_path(graph)
                results.append((removed_count, largest_cc_size, avg_shortest_path))

    return results


def select_node(graph, strategy, nodes):
    """Select a node based on the strategy."""
    if strategy == "highest_degree":
        return max(graph.degree, key=lambda x: x[1])[0]
    elif strategy == "random":
        return random.choice(nodes)


def calculate_average_shortest_path(graph):
    """Calculate the average shortest path length, considering disconnected components."""
    if nx.is_connected(graph):
        return nx.average_shortest_path_length(graph)

    # If the graph is disconnected, calculate for each connected component and return the weighted average
    components = list(nx.connected_components(graph))
    component_lengths = []

    for component in components:
        subgraph = graph.subgraph(component)
        if nx.is_connected(subgraph):
            component_lengths.append(nx.average_shortest_path_length(subgraph))
        else:
            # If subgraph is still disconnected, we handle it by calculating distances for the subgraph
            # and taking a simple average (or a similar fallback for disconnected subgraphs).
            component_lengths.append(
                float('nan'))  # To ensure calculation, we can use a placeholder for disconnected subgraphs.

    # Return the weighted average of the component lengths
    valid_lengths = [length for length in component_lengths if
                     not isinstance(length, float) or not (length != length)]  # filter out NaN
    if valid_lengths:
        return sum(valid_lengths) / len(valid_lengths)
    else:
        return float('nan')


def plot_results(results_ba, results_random, subplot_index, row_offset):
    """Plot results for Barabási-Albert and Random graphs."""
    # Ensure results are aligned by the number of nodes removed
    ba_removed_count, ba_largest_cc_sizes, ba_avg_shortest_paths = zip(*results_ba)
    random_removed_count, random_largest_cc_sizes, random_avg_shortest_paths = zip(*results_random)

    # x_values are the number of nodes removed
    x_values = list(ba_removed_count)

    # Top row: Relative size
    plt.subplot(2, 2, subplot_index + row_offset)
    plt.plot(x_values, ba_largest_cc_sizes, label="BA Graph", color='blue')
    plt.plot(x_values, random_largest_cc_sizes, label="Random Graph", color='orange')
    plt.title("Relative Size", fontsize=12)
    plt.xlabel("Nodes Removed")
    plt.ylabel("Relative Size")
    plt.legend()

    # Bottom row: Average path length
    plt.subplot(2, 2, subplot_index + row_offset + 2)
    plt.plot(x_values, ba_avg_shortest_paths, label="BA Graph", color='blue')
    plt.plot(x_values, random_avg_shortest_paths, label="Random Graph", color='orange')
    plt.title("Avg Path Length", fontsize=12)
    plt.xlabel("Nodes Removed")
    plt.ylabel("Path Length")
    plt.legend()


def setup_plot():
    """Prepare the figure and axes for the plots."""
    plt.figure(figsize=(14, 10))  # Set the figure size (width, height)


def save_plot():
    """Save the plot to a file."""
    save_path = "../results/tas10/"
    file_name = "task10.png"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + file_name)


def main():
    n = 10_000
    m = 2
    p = 0.015

    # Generate initial graphs for Barabási-Albert and Random graphs
    initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
    initial_graph = generate_connected_graph(initial_node_count)

    # Random Graph and Barabási-Albert Graph
    g_random = nx.Graph(generate_random_graph(n, p))
    g_ba = create_barabasi_albert_graph(initial_graph, m, n)

    # parallelize the computation for both graphs
    with ThreadPoolExecutor() as executor:
        future_results = {  # submit tasks to the executor
            "random_high_deg": executor.submit(remove_nodes, g_random.copy(), "highest_degree", "Random Graph"),
            "random_random": executor.submit(remove_nodes, g_random.copy(), "random", "Random Graph"),
            "ba_high_deg": executor.submit(remove_nodes, g_ba.copy(), "highest_degree", "Barabási-Albert Graph"),
            "ba_random": executor.submit(remove_nodes, g_ba.copy(), "random", "Barabási-Albert Graph")
        }

        # Wait for all results to be computed
        random_high_deg_results = future_results["random_high_deg"].result()
        random_random_results = future_results["random_random"].result()
        ba_high_deg_results = future_results["ba_high_deg"].result()
        ba_random_results = future_results["ba_random"].result()

    # prepare the plot and save the results
    setup_plot()
    plot_results(ba_high_deg_results, random_high_deg_results, 1, 0)
    plot_results(ba_random_results, random_random_results, 2, 0)
    save_plot()


if __name__ == "__main__":
    main()
