import csv


def read_csv(file_path):
    result = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            result.append(row)
    return result


def main():
    print("Hello, World!")
    data = read_csv("../datasets/KarateClub.csv")
    print(data)


if __name__ == "__main__":
    main()
