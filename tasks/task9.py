import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import networkx as nx
import random
from task7 import create_barabasi_albert_graph, generate_connected_graph, STARTING_NODES_COUNT_RANGE, \
    check_graph_properties
from task8 import print_graph_info
from task6 import generate_random_graph


def generate_data():
    np.random.seed(42)
    normal_data = np.random.normal(loc=5, scale=2, size=1000)
    exponential_data = np.random.exponential(scale=3, size=1000)
    power_law_data = np.random.pareto(a=2.5, size=1000)
    poisson_data = np.random.poisson(lam=3, size=1000)
    lognormal_data = np.random.lognormal(mean=1, sigma=0.5, size=1000)

    return {
        "normal": normal_data,
        "exponential": exponential_data,
        "power_law": power_law_data,
        "poisson": poisson_data,
        "lognormal": lognormal_data
    }


def fit_distributions(data_dict):
    fitted_distributions = {}

    for key, data in data_dict.items():
        if key == 'normal':
            params = stats.norm.fit(data)
            fitted_distributions[key] = ('norm', params)
        elif key == 'exponential':
            params = stats.expon.fit(data)
            fitted_distributions[key] = ('expon', params)
        elif key == 'power_law':
            params = stats.powerlaw.fit(data, floc=0)  # Fixed location parameter
            fitted_distributions[key] = ('powerlaw', params)
        elif key == 'poisson':
            lambda_estimate = np.mean(data)
            fitted_distributions[key] = ('poisson', lambda_estimate)
        elif key == 'lognormal':
            params = stats.lognorm.fit(data)
            fitted_distributions[key] = ('lognorm', params)

    return fitted_distributions


def plot_distributions(data_dict, fitted_distributions):
    x = np.linspace(-5, 15, 1000)  # x values for the fitted distributions

    for key, data in data_dict.items():
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label=f'{key} data')

        dist_name, params = fitted_distributions[key]
        if dist_name == 'norm':
            plt.plot(x, stats.norm.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'expon':
            plt.plot(x, stats.expon.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'powerlaw':
            plt.plot(x, stats.powerlaw.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'poisson':
            x_poisson = np.arange(0, 15)
            plt.plot(x_poisson, stats.poisson.pmf(x_poisson, params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'lognorm':
            plt.plot(x, stats.lognorm.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')

        plt.title(f'{key.capitalize()} Distribution')
        plt.legend()
        plt.show()


def main():
    data_dict = generate_data()
    fitted_distributions = fit_distributions(data_dict)
    plot_distributions(data_dict, fitted_distributions)
    for key, value in fitted_distributions.items():
        print(f'{key.capitalize()} distribution: {value}\n')

    n = 5_500  # Number of nodes
    m = 2  # Number of edges to attach from a new node to existing nodes
    p = 0.1  # Probability of rewiring each edge

    initial_node_count = random.randint(*STARTING_NODES_COUNT_RANGE)
    initial_graph = generate_connected_graph(initial_node_count)
    print_graph_info(initial_graph, "Initial")

    G_BA = create_barabasi_albert_graph(initial_graph, m, n)
    print_graph_info(G_BA, "Barabasi-Albert")

    random_graph_edges = generate_random_graph(n, p)
    G_RND = nx.Graph(random_graph_edges)
    print_graph_info(G_RND, "Random")


if __name__ == '__main__':
    main()
