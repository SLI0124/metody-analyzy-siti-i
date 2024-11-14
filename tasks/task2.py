import task1
from tasks.task1 import format_data, NUMBER_OF_NODES
from sys import maxsize

INF = maxsize


def read_and_format_data(file_path):
    raw_data = task1.read_csv(file_path)
    formatted_data = format_data(raw_data)
    return formatted_data


def initialize_adjacency_matrix(formatted_data):
    adjacency_matrix = task1.get_adjacency_matrix(formatted_data)
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i == j:
                adjacency_matrix[i][j] = 0
            elif adjacency_matrix[i][j] == 0:
                adjacency_matrix[i][j] = INF
    return adjacency_matrix


def floyd_warshall(adjacency_matrix):
    for k in range(NUMBER_OF_NODES):
        for i in range(NUMBER_OF_NODES):
            for j in range(NUMBER_OF_NODES):
                if adjacency_matrix[i][j] > adjacency_matrix[i][k] + adjacency_matrix[k][j]:
                    adjacency_matrix[i][j] = adjacency_matrix[i][k] + adjacency_matrix[k][j]
    return adjacency_matrix


def calculate_diameter_and_mean_distance(adjacency_matrix):
    diameter = 0
    total_distance = 0
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if adjacency_matrix[i][j] > diameter:
                diameter = adjacency_matrix[i][j]
            total_distance += adjacency_matrix[i][j]
    mean_average_distance = total_distance / (NUMBER_OF_NODES * (NUMBER_OF_NODES - 1))
    return diameter, mean_average_distance


def calculate_closeness_centrality(adjacency_matrix):
    closeness_centrality = []
    for i in range(NUMBER_OF_NODES):
        sum_of_shortest_paths = sum(adjacency_matrix[i])
        closeness_centrality.append(NUMBER_OF_NODES / sum_of_shortest_paths)
    return closeness_centrality


def main():
    formatted_data = read_and_format_data("../datasets/KarateClub.csv")
    adjacency_matrix = initialize_adjacency_matrix(formatted_data)

    print("\nAdjacency matrix:")
    for idx, row in enumerate(adjacency_matrix):
        print(f"{idx + 1}: {row}")

    adjacency_matrix = floyd_warshall(adjacency_matrix)

    print("\nShortest path matrix:")
    for idx, row in enumerate(adjacency_matrix):
        print(f"{idx + 1}: {row}")

    diameter, mean_average_distance = calculate_diameter_and_mean_distance(adjacency_matrix)
    print(f"\nDiameter: {diameter}")
    print(f"Mean average distance: {mean_average_distance}\n")

    closeness_centrality = calculate_closeness_centrality(adjacency_matrix)
    print("Closeness centrality:")
    for idx, value in enumerate(closeness_centrality):
        print(f"{idx + 1}: {value}")


if __name__ == "__main__":
    main()
