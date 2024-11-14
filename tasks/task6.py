import random
import csv
import os

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


def save_to_csv(edges, n, filename_prefix):
    file_prefix = "../results/task6/"

    if not os.path.exists(os.path.dirname(file_prefix)):
        os.makedirs(os.path.dirname(file_prefix))

    # Find all nodes present in edges and ensure all nodes up to `n` are included
    all_nodes = set(range(n))

    # Save edges to CSV
    with open(file_prefix + filename_prefix + "_edges.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(edges)

    # Save nodes to CSV
    with open(file_prefix + filename_prefix + "_nodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        for node in all_nodes:
            writer.writerow([node])


def main():
    n = 550
    p_values = [0.001, 0.0059, 0.01]

    for p in p_values:
        edges = generate_graph(n, p)
        plot_name = f"Graph with n={n} and p={p}"
        print(plot_name)
        print(f"Number of edges: {len(edges)}")
        print(f"Multi edges: {has_graph_multi_edges(edges)}")
        print(f"Loops: {has_graph_loops(edges)}")

        if has_graph_multi_edges(edges) or has_graph_loops(edges):
            print("Graph has multi-edges or loops, rerun the generation")
            continue
        else:
            print("Graph does not have multi-edges or loops, saving to CSV")
            save_to_csv(edges, n, f"graph_n_{n}_p_{p}")
        print()


if __name__ == '__main__':
    main()
