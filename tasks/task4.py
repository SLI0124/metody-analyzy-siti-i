import os

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge  # Import Wedge directly

from tasks.task1 import read_csv, format_data


def get_louvain_communities(G):
    return nx.algorithms.community.louvain_communities(G)


def get_label_propagation_communities(G):
    return nx.algorithms.community.label_propagation_communities(G)


def get_kernighan_lin_bisection(G):
    # bisect the graph to obtain 2 communities
    partition = nx.algorithms.community.kernighan_lin_bisection(G)

    # further bisect each community to obtain 4 communities
    communities = []
    for subgraph_nodes in partition:
        # create a subgraph for each partition and apply bisection again
        subgraph = G.subgraph(subgraph_nodes)
        sub_partition = nx.algorithms.community.kernighan_lin_bisection(subgraph)
        communities.extend(sub_partition)

    return communities


def get_girvan_newman(G, k=4):
    generator = nx.algorithms.community.girvan_newman(G)
    cut_level = k - 1
    communities = None
    for i in range(cut_level):
        communities = next(generator)

    return communities  # return the last communities


def get_k_clique_communities(G, k=4):
    result = []
    for community in nx.algorithms.community.k_clique_communities(G, k):
        result.append(set(community))

    return result


def calculate_modularity(G, communities):
    return nx.algorithms.community.modularity(G, communities)


def visualize_communities(G, communities, method_name):
    """https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html"""
    pos = nx.spring_layout(G)
    plt.figure()
    plt.title(f"{method_name} Communities")

    # mapping for each node to it corresponding communities
    node_community = {node: [] for node in G.nodes()}
    for i, community in enumerate(communities):
        for node in community:
            node_community[node].append(i)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    pie_colors = ['lightgreen', 'lightblue', 'lightpink', 'yellow']

    pie_size = 0.0666
    for node, community_indices in node_community.items():
        community_counts = [0] * len(communities)  # calculate the number of communities for each node
        for idx in community_indices:  # if a node belongs to multiple communities
            community_counts[idx] += 1

        total_count = sum(community_counts)
        proportions = [count / total_count if total_count > 0 else 0 for count in
                       community_counts]  # correction for division by zero

        block_percentage = []  # list of wedges/pie blocks to draw
        start_angle = 90  # Start the pie chart line at 90 degrees, meaning we start at 12 o'clock
        for i, proportion in enumerate(proportions):
            if proportion > 0:  # if the node belongs to the multiple communities
                angle = proportion * 360
                block_percentage.append(Wedge((pos[node][0], pos[node][1]),
                                              pie_size,
                                              start_angle,
                                              start_angle - angle,
                                              color=pie_colors[i]))
                start_angle -= angle

        # Add pies to the axes
        for block in block_percentage:
            plt.gca().add_patch(block)

    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
    plt.axis('equal')  # makes sure the pie chart is drawn as a circle

    save_dir = f"../results/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/task4_{method_name.replace(' ', "_")}_communities.png")


def print_formatted_communities(method_name, communities, modularity=None):
    print(f"{method_name}:\n"
          f"Total communities: {len(communities)}\n"
          f"Modularity: {modularity}\n"
          f"Communities: {communities}\n")


def main():
    data = read_csv("../datasets/KarateClub.csv")
    formatted_data = format_data(data)
    G = nx.Graph()
    G.add_edges_from(formatted_data)

    methods_dict = {
        "Louvain": get_louvain_communities,
        "Label propagation": get_label_propagation_communities,
        "Kernighan Lin Bisection": get_kernighan_lin_bisection,
        "Girvan Newman": get_girvan_newman,
        "K-clique": get_k_clique_communities
    }

    for method_name, method in methods_dict.items():
        communities = method(G)
        if method_name == "K-clique":
            modularity = "N/A"
        else:
            modularity = calculate_modularity(G, communities)
        print_formatted_communities(method_name, communities, modularity)
        visualize_communities(G, communities, method_name)


if __name__ == '__main__':
    main()
