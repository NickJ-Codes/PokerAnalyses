from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from clean_and_load_data import get_clean_data
from cluster_config import (
    FILE_TRAINING_DATA,
    FILE_NEWDATA,
    CLUSTERING_FEATURES,
    EXCLUDE_CLUSTERING_FEATURES,
    N_CLUSTERS
)

def explore_data(df):
    """
    Perform exploratory data analysis on the cleaned dataset
    """
    print("\nDataset Info:")
    print(df.info())

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nBasic statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

def prepare_features(df):
    """
    Prepare features for clustering analysis
    """
    try:
        # Select numerical columns for clustering
        feature_cols = [col for col in CLUSTERING_FEATURES if col not in EXCLUDE_CLUSTERING_FEATURES]

        print("\nFeatures selected for clustering:")
        print(feature_cols)

        # Create feature matrix
        X = df[feature_cols]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, feature_cols

    except Exception as e:
        print(f"Error preparing features: {e}")
        return None, None

def find_optimal_k(X_scaled):
    """
    Find optimal number of clusters using elbow method and silhouette score
    """
    inertias = []
    silhouette_scores_list = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k,
                        random_state=42,
                        n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores_list.append(silhouette_score(X_scaled, kmeans.labels_))

    # Plot elbow curve
    plt.figure(figsize=(12, 5))

    # Inertia plot
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')

    # Silhouette score plot
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores_list, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs k')

    plt.tight_layout()
    plt.show()

    return inertias, silhouette_scores_list

def perform_clustering(X_scaled, n_clusters):
    """
    Perform KMeans clustering with the specified number of clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return cluster_labels

def analyze_clusters(df, cluster_labels, feature_cols):
    """
    Analyze the characteristics of each cluster
    """
    # Add cluster labels to the original dataframe
    df['Cluster'] = cluster_labels

    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster')[feature_cols].mean()
    print("\nCluster Centers (mean values):")
    print(cluster_stats)

    return df

def main():
    # Load and clean data

    # Hardcoded - select 1 for training file or 2 for new data file
    selected_file_number = 1
    if selected_file_number == 1:
        file_selected = FILE_TRAINING_DATA
        print("Training file selected")
    elif selected_file_number == 2:
        file_selected = FILE_NEWDATA
        print("New data file selected")

    df = get_clean_data(file_selected)

    if df is not None:
        # Perform exploratory analysis
        explore_data(df)

        # Prepare features for clustering
        X_scaled, feature_cols = prepare_features(df)

        if X_scaled is not None:
            # Find optimal number of clusters
            inertias, silhouette_scores = find_optimal_k(X_scaled)

            # Perform clustering
            cluster_labels = perform_clustering(X_scaled, N_CLUSTERS)

            # Analyze clusters
            df_with_clusters = analyze_clusters(df, cluster_labels, feature_cols)

if __name__ == "__main__":
    main()
