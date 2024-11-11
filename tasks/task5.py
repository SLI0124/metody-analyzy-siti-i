import numpy as np
import pandas as pd
import networkx as nx
import csv


def load_csv(file_path, delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    # Replace commas in numeric columns and convert them to float, excluding the last column
    numeric_data = data.iloc[:, :-1].replace(',', '.', regex=True).astype(float)
    return numeric_data, data.iloc[:, -1]  # Return numeric data and the non-numeric "Species" column


def gaussian_similarity(x, y, sigma=1.0):
    distance = np.linalg.norm(x - y)  # Euclidean distance
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))  # Gaussian similarity


def build_knn_graph(data, k=3, sigma=1.0):
    G = nx.Graph()
    for i, point in enumerate(data):
        distances = [(j, gaussian_similarity(point, other, sigma))
                     for j, other in enumerate(data) if i != j]  # Calculate similarity with all other points
        distances.sort(key=lambda x: -x[1])  # Sort by similarity in descending order
        for j, sim in distances[:k]:  # Select top k neighbors
            G.add_edge(i, j, weight=sim)
    return G


def build_e_radius_graph(data, epsilon=0.9, sigma=1.0):
    G = nx.Graph()
    for i, point in enumerate(data):  # For each point
        for j, other in enumerate(data):  # For each other point
            if i != j:  # If the points are different
                sim = gaussian_similarity(point, other, sigma)  # Calculate similarity
                if sim > epsilon:  # If similarity is greater than epsilon
                    G.add_edge(i, j, weight=sim)  # Add edge to graph
    return G


def build_combined_graph(data, k=3, epsilon=0.9, sigma=1.0):
    G = nx.Graph()
    for i, point in enumerate(data):
        # kNN criteria
        distances = [(j, gaussian_similarity(point, other, sigma))
                     for j, other in enumerate(data) if i != j]
        distances.sort(key=lambda x: -x[1])
        for j, sim in distances[:k]:
            G.add_edge(i, j, weight=sim)

        # e-radius criteria
        for j, other in enumerate(data):
            if i != j:
                sim = gaussian_similarity(point, other, sigma)
                if sim > epsilon:
                    G.add_edge(i, j, weight=sim)
    return G


def save_graph_to_csv(G, file_name):
    # add to prefix path "../results/task5_" to save the file in the correct directory
    with open(f"../results/task5_{file_name}", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Sim"])
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data['weight']])


def main():
    file_path = "../datasets/iris.csv"
    numeric_data, species = load_csv(file_path, delimiter=';')
    numeric_data = numeric_data.values  # Convert DataFrame to numpy array for processing

    # Parameters
    k = 3
    sigma = 1.0
    epsilon = 0.9

    # Create graphs
    knn_graph = build_knn_graph(numeric_data, k=k, sigma=sigma)
    e_radius_graph = build_e_radius_graph(numeric_data, epsilon=epsilon, sigma=sigma)
    combined_graph = build_combined_graph(numeric_data, k=k, epsilon=epsilon, sigma=sigma)

    # Save graphs to CSV
    save_graph_to_csv(knn_graph, f"knn_graph_k_{k}_sigma_{sigma}.csv")
    save_graph_to_csv(e_radius_graph, f"e_radius_graph_epsilon_{epsilon}_sigma_{sigma}.csv")
    save_graph_to_csv(combined_graph, f"combined_graph_k_{k}_epsilon_{epsilon}_sigma_{sigma}.csv")


if __name__ == "__main__":
    main()
