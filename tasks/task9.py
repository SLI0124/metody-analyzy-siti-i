import random
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import networkx as nx

from task7 import create_barabasi_albert_graph, generate_connected_graph, STARTING_NODES_COUNT_RANGE
from task8 import print_graph_info
from task6 import generate_random_graph

SIZE = 1_000


def generate_data():
    np.random.seed(42)
    normal_data = np.random.normal(loc=5, scale=2, size=SIZE)
    exponential_data = np.random.exponential(scale=3, size=SIZE)
    power_law_data = np.random.pareto(a=2.5, size=SIZE)
    poisson_data = np.random.poisson(lam=3, size=SIZE)
    lognormal_data = np.random.lognormal(mean=1, sigma=0.5, size=SIZE)

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


def plot_distributions_subplots(data_dict, fitted_distributions, save_path):
    rows, cols = 3, 2  # grid layout for the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for idx, (key, data) in enumerate(data_dict.items()):
        ax = axes[idx]
        ax.hist(data, bins=30, density=True, alpha=0.6, color='g', label=f'{key} data')

        dist_name, params = fitted_distributions[key]
        if dist_name == 'norm':
            x = np.linspace(np.min(data) - 1, np.max(data) + 1, 1000)
            ax.plot(x, stats.norm.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'expon':
            x = np.linspace(0, np.max(data) + 1, 1000)
            ax.plot(x, stats.expon.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'powerlaw':
            x = np.linspace(0, np.max(data) + 1, 1000)
            ax.plot(x, stats.powerlaw.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'poisson':
            x_poisson = np.arange(0, np.max(data) + 1)
            ax.plot(x_poisson, stats.poisson.pmf(x_poisson, params), 'r-', label=f'Fitted {dist_name}')
        elif dist_name == 'lognorm':
            x = np.linspace(np.min(data) - 1, np.max(data) + 1, 1000)
            ax.plot(x, stats.lognorm.pdf(x, *params), 'r-', label=f'Fitted {dist_name}')

        ax.set_title(f'{key.capitalize()} Distribution')
        ax.legend()

    for idx in range(len(data_dict), rows * cols):  # Hide any unused subplots
        axes[idx].axis('off')

    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, 'fitted_distributions.png'))


def plot_linear_and_log_degree_distributions(G_BA, G_RND, save_path):
    BA_degrees = [degree for node, degree in G_BA.degree()]
    BA_degree_counts = np.bincount(BA_degrees)
    BA_degree_counts = BA_degree_counts[BA_degree_counts > 0]

    RND_degrees = [degree for node, degree in G_RND.degree()]
    RND_degree_counts = np.bincount(RND_degrees)
    RND_degree_counts = RND_degree_counts[RND_degree_counts > 0]

    # create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # plot degree distribution for Barabasi-Albert graph
    axes[0, 0].plot(BA_degree_counts, 'b-', marker='o')
    axes[0, 0].set_title('Degree Distribution (Linear Scale) - Barabasi-Albert')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].plot(BA_degree_counts, 'b-', marker='o')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('Degree Distribution (Log-Log Scale) - Barabasi-Albert')

    # plot degree distribution for Random graph
    axes[1, 0].plot(RND_degree_counts, 'r-', marker='o')
    axes[1, 0].set_title('Degree Distribution (Linear Scale) - Random')
    axes[1, 0].set_xlabel('Degree')
    axes[1, 0].set_ylabel('Count')

    axes[1, 1].plot(RND_degree_counts, 'r-', marker='o')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_title('Degree Distribution (Log-Log Scale) - Random')

    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, 'degree_distributions.png'))


def plot_cdf_ccdf(G_BA, G_RND, save_path):
    BA_degrees = [degree for node, degree in G_BA.degree()]
    BA_degrees = np.sort(BA_degrees)

    cdf_BA = np.arange(1, len(BA_degrees) + 1) / len(BA_degrees)
    ccdf_BA = 1 - cdf_BA

    RND_degrees = [degree for node, degree in G_RND.degree()]
    RND_degrees = np.sort(RND_degrees)

    cdf_RND = np.arange(1, len(RND_degrees) + 1) / len(RND_degrees)
    ccdf_RND = 1 - cdf_RND

    # plot four subplots in 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # plot CDF for Barabasi-Albert graph
    axes[0, 0].plot(BA_degrees, cdf_BA, 'b-', marker='o')
    axes[0, 0].set_title('CDF - Barabasi-Albert')

    # plot CCDF for Barabasi-Albert graph
    axes[0, 1].plot(BA_degrees, ccdf_BA, 'b-', marker='o')
    axes[0, 1].set_title('CCDF - Barabasi-Albert')

    # plot CDF for Random graph
    axes[1, 0].plot(RND_degrees, cdf_RND, 'r-', marker='o')
    axes[1, 0].set_title('CDF - Random')

    # plot CCDF for Random graph
    axes[1, 1].plot(RND_degrees, ccdf_RND, 'r-', marker='o')
    axes[1, 1].set_title('CCDF - Random')

    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, 'cdf_ccdf.png'))


def fit_and_plot_ccdf(graph, title, subplot_index):
    """
    Fit and plot CCDF with Poisson, Normal, Exponential, and Power-law fits.

    Args:
        graph: NetworkX graph object.
        title: Title for the plot.
        subplot_index: Subplot index for positioning.
    """
    # Compute degrees and sort them
    degrees = np.array([d for _, d in graph.degree()])
    sorted_degrees = np.sort(degrees)

    # Calculate CCDF
    ccdf = 1 - (np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees))

    # Fit distributions
    # Poisson
    lambda_poisson = np.mean(degrees)
    poisson_ccdf = 1 - stats.poisson.cdf(sorted_degrees, mu=lambda_poisson)

    # Normal
    mu_normal, sigma_normal = stats.norm.fit(degrees)
    normal_ccdf = 1 - stats.norm.cdf(sorted_degrees, loc=mu_normal, scale=sigma_normal)

    # Exponential
    loc_expon, scale_expon = stats.expon.fit(degrees)
    expon_ccdf = 1 - stats.expon.cdf(sorted_degrees, loc=loc_expon, scale=scale_expon)

    # Power-law
    a_power, loc_power, scale_power = stats.powerlaw.fit(degrees, floc=0)
    power_ccdf = 1 - stats.powerlaw.cdf(sorted_degrees, a=a_power, loc=loc_power, scale=scale_power)

    # Plot CCDF
    plt.subplot(1, 2, subplot_index)
    plt.loglog(sorted_degrees, ccdf, 'o', alpha=0.7, label='Empirical CCDF')
    plt.loglog(sorted_degrees, poisson_ccdf, '-', label='Poisson Fit')
    plt.loglog(sorted_degrees, normal_ccdf, '-', label='Normal Fit')
    plt.loglog(sorted_degrees, expon_ccdf, '-', label='Exponential Fit')
    plt.loglog(sorted_degrees, power_ccdf, '-', label='Power-law Fit')

    plt.title(f"{title} - CCDF with Fits")
    plt.xlabel("Degree")
    plt.ylabel("CCDF")
    plt.legend()


def main():
    save_path = "../results/task9"
    # task 1
    data_dict = generate_data()
    fitted_distributions = fit_distributions(data_dict)
    plot_distributions_subplots(data_dict, fitted_distributions, save_path)
    for key, value in fitted_distributions.items():
        print(f'{key.capitalize()} distribution: {value}\n')

    # task 2
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

    # plot_linear_and_log_degree_distributions(G_BA, G_RND, save_path)
    # plot_cdf_ccdf(G_BA, G_RND, save_path)

    plt.figure(figsize=(15, 7))
    fit_and_plot_ccdf(G_BA, "Barabasi-Albert", 1)
    fit_and_plot_ccdf(G_RND, "Random", 2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ccdf_with_fits.png'))


if __name__ == '__main__':
    main()
