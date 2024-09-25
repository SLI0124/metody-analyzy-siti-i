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


def calculate_max_degree(adjacency_list):
    max_degree = 0
    node_index = 0
    for idx, row in enumerate(adjacency_list):
        degree = len(row)
        if degree > max_degree:
            max_degree = degree
            node_index = idx
    return max_degree, node_index


def calculate_min_degree(adjacency_list):
    min_degree = len(adjacency_list[0])
    node_index = 0
    for idx, row in enumerate(adjacency_list):
        degree = len(row)
        if degree < min_degree:
            min_degree = degree
            node_index = idx
    return min_degree, node_index


def calculate_avg_degree(adjacency_list):
    total_degree = 0
    for idx, row in enumerate(adjacency_list):
        total_degree += len(row)
    return total_degree / len(adjacency_list)


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

    max_degree, max_node_index = calculate_max_degree(adjacency_list)
    print(f"\nMax degree: {max_degree} with index: {max_node_index + 1}\n")

    min_degree, min_node_index = calculate_min_degree(adjacency_list)
    print(f"Min degree: {min_degree} with index: {min_node_index + 1}\n")

    avg_degree = calculate_avg_degree(adjacency_list)
    print(f"Avg degree: {avg_degree:.2f} ({avg_degree})\n")


if __name__ == "__main__":
    main()
