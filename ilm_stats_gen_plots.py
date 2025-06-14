from matplotlib import pyplot as plt

"""
Data for ILM statistics.

DATA[model_name][top_k_index] = accuracy
"""
# DATA = {
#     'Baseline': [
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0]
#     ],
#     'Identity-grad': [
#         [0.224, 0.4521, 0.8355, 0.8357],
#         [0.1197, 0.4222, 0.438, 0.7551],
#         [0.6654, 0.6722, 0.6738, 0.6761],
#         [0.1799, 0.3428, 0.3507, 0.5525]
#     ],
#     'Inv-first': [
#         [0.3946, 0.9839, 0.9992, 0.9993],
#         [0.0002, 0.0005, 0.0007, 0.0008],
#         [0.0051, 0.0171, 0.0294, 0.0341],
#         [0.0064, 0.0097, 0.0117, 0.0118]
#     ],
#     'Bert-like': [
#         [0.3258, 0.3417, 0.3429, 0.3871],
#         [0.1017, 0.2078, 0.3065, 0.3584],
#         [0.0013, 0.0021, 0.0023, 0.0033],
#         [0.0095, 0.019, 0.02, 0.0284]
#     ],
#     'Reverse bigram': [
#         [0.6043, 0.8417, 0.8739, 0.8684],
#         [0.2379, 0.2817, 0.9213, 0.9587],
#         [0.8492, 0.8929, 0.8997, 0.9029],
#         [0.2294, 0.3219, 0.8933, 0.9014]
#     ]
# }

DATA = {
    'Baseline': [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    'Identity-grad (random init)': [
        [0.02, 0.03, 0.04, 0.04],  # Token 0
        [0.24, 0.83, 1.01, 1.79],  # Token 1
        [0.31, 0.33, 0.33, 0.47],  # Token 2
        [0.16, 0.23, 0.25, 0.39],  # Token 19
        [0.16, 0.22, 0.24, 0.33],  # Token 24
        [0.13, 0.18, 0.20, 0.29]  # Token 29
    ],
    'Identity-grad (bigram init)': [
        [22.40, 45.22, 83.55, 83.57],  # Token 0
        [11.97, 42.22, 43.79, 75.48],  # Token 1
        [66.54, 67.22, 67.38, 67.61],  # Token 2
        [17.87, 29.95, 31.45, 32.60],  # Token 19
        [21.54, 34.53, 35.89, 36.94],  # Token 24
        [21.33, 32.75, 34.13, 35.11]  # Token 29
    ],
    'Reverse bigram': [
        [60.43, 84.17, 87.39, 87.52],  # Token 0
        [23.79, 28.17, 92.13, 95.87],  # Token 1
        [84.92, 89.29, 89.87, 90.29],  # Token 2
        [14.84, 30.78, 46.05, 50.77],  # Token 19
        [18.81, 35.98, 48.72, 52.94],  # Token 24
        [19.57, 35.69, 47.29, 52.08]  # Token 29
    ],
    'Bert-like (pad init)': [
        [32.58, 34.18, 34.30, 38.71],  # Token 0
        [10.17, 20.76, 30.64, 35.84],  # Token 1
        [0.13, 0.21, 0.23, 0.33],  # Token 2
        [2.26, 3.22, 3.50, 4.18],  # Token 19
        [3.26, 4.40, 4.79, 5.55],  # Token 24
        [3.27, 4.51, 4.92, 5.72]  # Token 29
    ],
    'Inv-first (pad init)': [
        [39.48, 98.39, 99.92, 99.93],  # Token 0
        [0.02, 0.05, 0.07, 0.08],  # Token 1
        [0.51, 1.71, 2.94, 3.42],  # Token 2
        [2.40, 4.03, 4.65, 4.77],  # Token 19
        [3.99, 5.93, 6.51, 6.66],  # Token 24
        [3.85, 5.61, 6.24, 6.36]  # Token 29
    ],
}


def plot_data(title: str, data: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(5, 3))
    plotted_points = []
    for label, values in data.items():
        top_k_indices = [i + 1 for i in range(len(values))]  # Top-K indices start from 1
        line, = ax.plot(top_k_indices, values, marker='o', label=label)
        plotted_points.append(line)

    ax.set_title(title)

    if title == 'Legend':
        ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), ncol=2)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for v in ax.spines.values():
            v.set_visible(False)
        for point in plotted_points:
            point.set_visible(False)
    else:
        ax.set_xticks(top_k_indices)
        ax.set_xticklabels([f'Top-{k}' for k in top_k_indices])
        ax.set_xlabel('Top-K')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        plt.grid()

    plt.tight_layout()
    filename = title.replace(" ", "_").replace("#", "").lower() + '.png'
    print(f'Saving plot to {filename}')
    fig.savefig(filename, dpi=300)


def main():
    # Prepare data for plotting
    for i in range(len(DATA['Baseline'])):
        t_i = [0, 1, 2, 19, 24, 29][i] + 1  # Token indices start from 1 in the plot
        plot_data(f'Inverse LM predicting #{t_i} token', {
            model: [DATA[model][i][j] for j in range(len(DATA[model][i]))]
            for model in DATA
        })

    i = 0
    plot_data('Legend', {
        model: [DATA[model][i][j] for j in range(len(DATA[model][i]))]
        for model in DATA
    })


if __name__ == '__main__':
    main()
