import os

import networkx as nx
import matplotlib.pyplot as plt

from tasks.task1 import read_csv, format_data


def get_clustering_coefficient(G):
    return nx.clustering(G)


def get_transitivity(G):
    return nx.transitivity(G)


def plot_clustering_coefficient_distribution(G):
    """
    on x-axis: degree of the node, on y-axis: average clustering coefficient of the node
    """
    degree_clustering = {}
    for node in G.nodes():  # iterate over all nodes
        degree = G.degree(node)  # get the degree of the node
        clustering = nx.clustering(G, node)  # get the clustering coefficient of the node
        if degree not in degree_clustering:  # if the degree is not in the dictionary, add it
            degree_clustering[degree] = []  # add the degree as key and an empty list as value
        degree_clustering[degree].append(clustering)  # append the clustering coefficient to the list

    # calculate the average clustering coefficient for each degree
    for degree, clustering_list in degree_clustering.items():
        degree_clustering[degree] = sum(clustering_list) / len(clustering_list)

    # sort the dictionary by key
    degrees = list(degree_clustering.keys())
    avg_clustering_values = list(degree_clustering.values())

    # plot the degree and average clustering coefficient
    plt.scatter(degrees, avg_clustering_values)
    # y range from 0 to 1 with step 0.25
    plt.yticks([i / 4 for i in range(5)])
    # set x ranges by 5
    plt.xticks([i for i in range(0, max(degrees) + 1, 5)])
    plt.xlabel("d (Degree of the node)")
    plt.ylabel("avg_CC (Average clustering coefficient)")
    plt.grid(True)

    # save the plot
    save_path = "../results/task3_degree_clustering_coefficient_distribution.png"

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.savefig(save_path)


def save_attributes_to_csv(G):
    """
    ID of the node, its degree, closeness centrality, and clustering coefficient.
    """
    clustering_coefficient = nx.clustering(G)
    closeness_centrality = nx.closeness_centrality(G)

    output_path = "../results/csv_result.csv"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write("ID, Degree, Closeness centrality, Clustering coefficient\n")
        for node in sorted(G.nodes(), key=int):  # sort the nodes by their ID in ascending order
            f.write(f"{node}, {G.degree(node)}, {closeness_centrality[node]}, {clustering_coefficient[node]}\n")


if __name__ == '__main__':
    data = read_csv("../datasets/KarateClub.csv")
    formatted_data = format_data(data)

    G = nx.Graph()
    G.add_edges_from(formatted_data)

    clustering_coefficient = get_clustering_coefficient(G)

    transitivity = get_transitivity(G)

    # transform keys into int
    clustering_coefficient = {int(key): value for key, value in clustering_coefficient.items()}
    sorted_clustering_coefficient = sorted(clustering_coefficient.items(), key=lambda x: x[0])

    for node, clustering in sorted_clustering_coefficient:
        print(f"Node {node} : clustering coefficient: {clustering}")

    print(f"\nTransitivity: {transitivity}")

    plot_clustering_coefficient_distribution(G)

    save_attributes_to_csv(G)
