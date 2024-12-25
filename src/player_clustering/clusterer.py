"""
Several goals

Goal 1: Identify clusters of players that are most common
"""

from data_clean import get_clean_data
from sklearn.cluster import KMeans
import seaborn as sns
import os
import matplotlib.pyplot as plt
from cluster_config import (
    filename_AllPlayers,
    N_CLUSTERS,
    CLUSTERING_FEATURES,
    RANDOM_STATE,
    MAX_ITERATIONS,
    CLUSTER_NAMES,
    CLUSTER_COLORS,
    N_INIT,
    OUTPUT_DIR,
    OUTPUT_FILE_PLAYERS,
    OUTPUT_FILE_CLUSTER_MEANS
)

def analyze_clusters(dfMain):
    """Print analysis of the clusters"""
    print("\nCluster Analysis:")
    print("-" * 50)
    for cluster_id in range(N_CLUSTERS):
        cluster_name = CLUSTER_NAMES[cluster_id]
        cluster_size = (dfMain['cluster'] == cluster_id).sum()
        # print(f"\nCluster {cluster_id} ({cluster_name}):")
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {cluster_size} players ({cluster_size/len(dfMain)*100:.1f}%)")
        print("Average values:")
        for feature in CLUSTERING_FEATURES:
            mean_value = dfMain[dfMain['cluster'] == cluster_id][feature].mean()
            print(f"- {feature}: {mean_value:.3f}")

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
    # Read your data
    dfMain = get_clean_data(filename_AllPlayers)  # adjust path as needed

    # Remove rows with missing values
    dfMain = dfMain.dropna()

    # Verify all features exist in the dataset
    missing_features = [col for col in CLUSTERING_FEATURES if col not in dfMain.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")

    # Extract features for clustering
    X = dfMain[CLUSTERING_FEATURES]

    # Initialize and fit KMeans
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITERATIONS,
        n_init=N_INIT
    )

    # Fit the model and predict clusters
    dfMain['cluster'] = kmeans.fit_predict(X)
    dfMain['cluster_name'] = dfMain['cluster'].map(CLUSTER_NAMES)

    # Create cluster means dataframe with both cluster number, name, and count of players
    cluster_means = dfMain.groupby(['cluster', 'cluster_name']).agg({
        **{feature: 'mean' for feature in CLUSTERING_FEATURES},
        'cluster': 'count'  # This adds the count of players
    }).round(3)

    # Rename the count column to something more descriptive
    cluster_means = cluster_means.rename(columns={'cluster': 'player_count'})

    # If you want to reset the index to make cluster and cluster_name regular columns
    cluster_means = cluster_means.reset_index()

    # Save clustered player data
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Save the classified dataframe
    try:
        dfMain.to_csv(OUTPUT_FILE_PLAYERS, index=False)
        cluster_means.to_csv(OUTPUT_FILE_CLUSTER_MEANS, index=True)
        print(f"Successfully saved classified data to {OUTPUT_FILE_PLAYERS}")
    except PermissionError:
        print("Permission denied. Please ensure:")
        print("1. The output directory is not read-only")
        print("2. The output files are not open in another program")
        print("3. You have write permissions in the output directory")
    except Exception as e:
        print(f"Error saving output file(s): {str(e)}")

    # Analyze and visualize
    analyze_clusters(dfMain)
    create_cluster_visualizations(dfMain, CLUSTERING_FEATURES)

if __name__ == "__main__":
    main()
