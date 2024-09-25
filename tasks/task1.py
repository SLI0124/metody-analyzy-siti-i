import csv

NUMBER_OF_NODES = 34


def read_csv(file_path):
    result = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            result.append(row)
    return result


def format_data(data):
    result = []
    for row in data:
        split_row = row[0].split(";")
        result.append(split_row)
    return result


def get_adjacency_matrix(data):
    matrix = [[0 for _ in range(NUMBER_OF_NODES)] for _ in range(NUMBER_OF_NODES)]
    for row in data:
        node1 = int(row[0]) - 1
        node2 = int(row[1]) - 1
        matrix[node1][node2] = 1
        matrix[node2][node1] = 1
    return matrix


def get_adjacency_list(data):
    adjacency_list = [[] for _ in range(NUMBER_OF_NODES)]
    for row in data:
        node1 = int(row[0]) - 1
        node2 = int(row[1]) - 1
        adjacency_list[node1].append(node2)
        adjacency_list[node2].append(node1)
    return adjacency_list


def main():
    data = read_csv("../datasets/KarateClub.csv")
    formatted_data = format_data(data)
    adjacency_matrix = get_adjacency_matrix(formatted_data)

    print("\nAdjacency matrix:")
    for idx, row in enumerate(adjacency_matrix):
        print(f"{idx + 1}: {row}")

    print("\nAdjacency list:")
    adjacency_list = get_adjacency_list(formatted_data)
    for idx, row in enumerate(adjacency_list):
        print(f"{idx + 1}: {row}")


if __name__ == "__main__":
    main()
