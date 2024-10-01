import task1
from tasks.task1 import format_data, NUMBER_OF_NODES
from sys import maxsize

INF = maxsize


def main():
    raw_data = task1.read_csv("../datasets/KarateClub.csv")
    formatted_data = format_data(raw_data)

    adjacency_matrix = task1.get_adjacency_matrix(formatted_data)
    print("\nAdjacency matrix:")
    for idx, row in enumerate(adjacency_matrix):
        print(f"{idx + 1}: {row}")

    # fill the diagonal with 0s and the rest with infinity
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i == j:
                adjacency_matrix[i][j] = 0
            elif adjacency_matrix[i][j] == 0:
                adjacency_matrix[i][j] = INF

    # Floyd-Warshall algorithm
    for k in range(NUMBER_OF_NODES):
        for i in range(NUMBER_OF_NODES):
            for j in range(NUMBER_OF_NODES):
                if adjacency_matrix[i][j] > adjacency_matrix[i][k] + adjacency_matrix[k][j]:
                    adjacency_matrix[i][j] = adjacency_matrix[i][k] + adjacency_matrix[k][j]

    print("\nShortest path matrix:")
    for idx, row in enumerate(adjacency_matrix):
        print(f"{idx + 1}: {row}")

    # diameter and mean average distance
    diameter = 0
    mean_average_distance = 0
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if adjacency_matrix[i][j] > diameter:
                diameter = adjacency_matrix[i][j]
            mean_average_distance += adjacency_matrix[i][j]

    mean_average_distance = mean_average_distance / (NUMBER_OF_NODES * (NUMBER_OF_NODES - 1))
    print(f"\nDiameter: {diameter}", end="\n\n")
    print(f"Mean average distance: {mean_average_distance}", end="\n\n")

    # closeness centrality
    closeness_centrality = []
    for i in range(NUMBER_OF_NODES):
        sum_of_shortest_paths = sum(adjacency_matrix[i])
        closeness_centrality.append(NUMBER_OF_NODES / sum_of_shortest_paths)

    print("Closeness centrality:")
    for idx, value in enumerate(closeness_centrality):
        print(f"{idx + 1}: {value}")


if __name__ == "__main__":
    main()
