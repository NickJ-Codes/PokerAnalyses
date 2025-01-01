import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cluster_config import (
    CLUSTER_NAMES,
    CLUSTER_COLORS,
    OUTPUT_FILE_PLAYERS,
    CLUSTERING_FEATURES
)

def load_clustered_data(filepath):
    """Load the clustered data from a saved file"""
    return pd.read_csv(filepath)

def create_cluster_visualizations(dfMain, features_for_clustering):
    """Create multiple tiled visualizations of the clusters with profitability-based coloring"""

    plt.style.use('seaborn-v0_8-darkgrid')

    # Create color mapping based on BB/100
    # Define color boundaries for different profitability levels
    bounds = [-100, -20, -5, 0, 5, 20, 100]
    colors = ['darkred', 'red', 'orange', 'yellow', 'yellowgreen', 'green', 'darkgreen']
    norm = plt.Normalize(min(bounds), max(bounds))

    # region First set of plots, all density plots
    plt.figure(figsize=(7, 7))

    # First plot is PFR vs VPIP density
    plt.subplot(2, 2, 1)
    sns.kdeplot(data=dfMain, x='vpip', y='pfr',
                cmap='RdYlGn', fill=True,
                levels=20)
    plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)  # diagonal line
    plt.xlabel('VPIP')
    plt.ylabel('PFR')
    plt.title('VPIP vs PFR (Player Density)')

    # Second plot is PFR vs 3bet density
    plt.subplot(2, 2, 2)
    sns.kdeplot(data=dfMain, x='pfr', y='3bet_f',
                cmap='RdYlGn', fill=True,
                levels=20)
    plt.xlabel('PFR')
    plt.ylabel('3bet Frequency')
    plt.title('PFR vs 3bet (Player Density)')

    # Third plot is WTSD vs WSD density
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=dfMain, x='wtsd', y='wsd',
                cmap='RdYlGn', fill=True,
                levels=20)
    plt.xlabel('Went to Showdown')
    plt.ylabel('Won at Showdown')
    plt.title('WTSD vs WSD (Player Density)')

    plt.tight_layout()
    plt.show()
    # endregion

    # region Second set of plots
    # Create distribution plots for key metrics
    plt.figure(figsize=(7, 7))

    # by cluster, BB/100 distribution
    plt.subplot(2, 2, 1)
    for cluster in sorted(dfMain['cluster'].unique()):
        cluster_data = dfMain[dfMain['cluster'] == cluster]
        sns.kdeplot(data=cluster_data['bb_100'],
                    label = CLUSTER_NAMES[cluster],
                    color=CLUSTER_COLORS[cluster],
                    fill=True,
                    alpha=0.3)

    plt.title('BB/100 Distribution by Cluster')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.legend()

    # by cluster, VPIP distribution
    plt.subplot(2, 2, 2)
    for cluster in sorted(dfMain['cluster'].unique()):
        cluster_data = dfMain[dfMain['cluster'] == cluster]
        sns.kdeplot(data=cluster_data['vpip'],
                    label = CLUSTER_NAMES[cluster],
                    color=CLUSTER_COLORS[cluster],
                    fill=True,
                    alpha=0.3)

    plt.title('VPIP Distribution by Cluster')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # by cluster, PFR distribution
    plt.subplot(2, 2, 3)
    for cluster in sorted(dfMain['cluster'].unique()):
        cluster_data = dfMain[dfMain['cluster'] == cluster]
        sns.kdeplot(data=cluster_data['pfr'],
                    label = CLUSTER_NAMES[cluster],
                    color=CLUSTER_COLORS[cluster],
                    fill=True,
                    alpha=0.3)

    plt.title('PFR Distribution by Cluster')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # by cluster, 3B distribution
    plt.subplot(2, 2, 4)
    for cluster in sorted(dfMain['cluster'].unique()):
        cluster_data = dfMain[dfMain['cluster'] == cluster]
        sns.kdeplot(data=cluster_data['3bet_f'],
                    label = CLUSTER_NAMES[cluster],
                    color=CLUSTER_COLORS[cluster],
                    fill=True,
                    alpha=0.3)

    plt.title('3Bet Distribution by Cluster')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)



    plt.tight_layout()
    plt.show()
    # endregion

    # region Third 3d scatterplot of PFR vs VPIP vs BB100
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for cluster in sorted(dfMain['cluster'].unique()):
        cluster_data = dfMain[dfMain['cluster'] == cluster]
        ax.scatter(cluster_data['vpip'],
                   cluster_data['pfr'],
                   cluster_data['bb_100'],
                   c=CLUSTER_COLORS[cluster],
                   label=CLUSTER_NAMES[cluster],
                   alpha=0.6)

    ax.set_xlabel('VPIP')
    ax.set_ylabel('PFR')
    ax.set_zlabel('BB/100')
    plt.title('3D Cluster Visualization')
    plt.legend()
    plt.show()
    # endregion

def main():
    # Define the path to your clustered data
    data_path = OUTPUT_FILE_PLAYERS

    # load the data
    df = load_clustered_data(data_path)

    # Generate visualizations
    create_cluster_visualizations(df, features_for_clustering = CLUSTERING_FEATURES)

if __name__ == "__main__":
    main()