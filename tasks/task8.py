import random
from task7 import check_graph_properties, generate_connected_graph, create_barabasi_albert_graph, \
    STARTING_NODES_COUNT_RANGE


def random_node_sampling(G, sample_size):
    nodes = random.sample(list(G.nodes()), sample_size)  # randomly select a sample of nodes
    selected_nodes = set(nodes)  # set of selected nodes
    # select edges where both endpoints are in the sample set VS
    edges_in_sample = [(u, v) for u, v in G.edges() if u in selected_nodes and v in selected_nodes]
    subgraph = G.subgraph(selected_nodes).copy()  # create a subgraph from the selected nodes
    subgraph.add_edges_from(edges_in_sample)

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
    return subgraph


def print_graph_info(G, name):
    check_graph_properties(G, name)
    print(f"Number of nodes in the {name} graph: {G.number_of_nodes()}")
    print(f"Number of edges in the {name} graph: {G.number_of_edges()}\n")


def main():
    n = 5_000  # Number of nodes in the graph
    m = 2  # Number of edges to attach from a new node to existing nodes
    desired_size = int(n * 0.15)  # Desired size of the sampled graph

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

    # Perform random node sampling
    sampled_graph = random_node_sampling(G_BA, desired_size)
    print_graph_info(sampled_graph, "Random Node Sample")

    # Perform random edge sampling
    sampled_graph = random_edge_sampling(G_BA, desired_size)
    print_graph_info(sampled_graph, "Random Edge Sample")

    # Perform random walk sampling
    start_node = random.choice(list(G_BA.nodes()))
    sampled_graph = random_walk_sampling(G_BA, start_node, desired_size)
    print_graph_info(sampled_graph, "Random Walk Sample")


if __name__ == "__main__":
    main()
