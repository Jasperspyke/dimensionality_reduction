import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# for reproducibility
np.random.seed(43)

# set up some variance in the data, differing means, etc.
n = 500
mu1 = np.zeros(5)
mu2 = np.ones(5) * 5
mean_info = mu2[:2]
cov1 = np.eye(5) + 0.2 * np.ones((5, 5))
cov1 -= 0.5 * np.eye(5)
cov2 = np.eye(5) + 0.6 * np.ones((5, 5))
cov2 += 0.5 * np.eye(5)

# set up some variance in the data, differing means, etc.
matrix = np.random.randn(5, 5)
matrix = matrix.dot(matrix.T)
matrix += 0.05 * (np.random.randn(5, 5) - 0.5)
matrix = 0.5 * (matrix + matrix.T)
matrix = (matrix - matrix.mean()) / matrix.std() * 0.6 + 1

# Sample from the distributions
data1 = np.random.multivariate_normal(mu1, cov1, size=n)
data2 = np.random.multivariate_normal(mu2, matrix, size=n)


# Concatenate the two datasets along axis 0
data = np.concatenate((data1, data2), axis=0)
print(data.shape)

if __name__ == '__main__':
    data_df1 = pd.DataFrame(data1)
    data_df2 = pd.DataFrame(data2)

    # Add class labels to the DataFrames
    data_df1['class'] = 'Normal'
    data_df2['class'] = 'Abnormal'

    # Combine the DataFrames
    data_df = pd.concat([data_df1, data_df2])
    # Set the Seaborn style and color palette
    sns.set(style="darkgrid", color_codes=True)
    sns.set_palette("husl")
    fig = plt.gcf()
    fig.set_facecolor('black')

    plt.rcParams['legend.fontsize'] = 24
    # Visualize the pairwise relationships in the 5-dimensional datasets with customizations
    pairs = sns.pairplot(data=data_df, hue='class', corner=True, diag_kind="kde", markers=["o", "s"],
                     plot_kws={'s': 20, 'alpha': 0.8, 'edgecolor': 'k', 'linewidth': 0.5},
                     diag_kws={'color': 'red', 'fill': True})
    pairs.map_upper(sns.kdeplot, cmap='coolwarm', levels=8, shade=True, shade_lowest=False)

    # Adjust the spacing between subplots
    pairs.fig.subplots_adjust(hspace=0.1, wspace=0.1)

    plt.show()

