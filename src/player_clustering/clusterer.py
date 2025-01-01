import os
import pickle
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from cluster_config import (
    N_CLUSTERS,
    CLUSTERING_FEATURES,
    RANDOM_STATE,
    CLUSTER_NAMES,
    MODEL_DIR,
    MODEL_FILE
)

class PokerPlayerClusterer:
    def __init__(self, n_clusters: int = N_CLUSTERS):
        """
        Initialize the clusterer with specified number of clusters.

        Args:
            n_clusters (int): Number of clusters to form
        """
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init='auto'  # Optimization: Use the new default
        )
        self.cluster_names: Optional[Dict] = None
        self.clustering_features: Optional[List] = None

    def fit(self,
            df: pd.DataFrame,
            clustering_features: List[str] = CLUSTERING_FEATURES,
            cluster_names: Dict[int, str] = CLUSTER_NAMES) -> 'PokerPlayerClusterer':
        """
        Train the clustering model

        Args:
            df: Input dataFrame
            clustering_features: list of feature columns to use for clustering
            cluster_names: mapping of cluster number to cluster names

        Returns:
            self: The fitted clusterer
        """
        if not all(feature in df.columns for feature in clustering_features):
            raise ValueError(f"Missing features. Required: {clustering_features}")

        self.clustering_features = clustering_features
        self.cluster_names = cluster_names

        # Extract features for clustering - using numpy array for better performance
        X = df[clustering_features].to_numpy()

        # Fit the model
        self.kmeans.fit(X)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict Clusters for New Data

        Args:
            df: Input dataFrame

        Returns:
            DataFrame with cluster labels
        """
        if not all(feature in df.columns for feature in self.clustering_features):
            raise ValueError(f"Missing features. Required: {self.clustering_features}")

        # Extract features and predict - using numpy array for better performance
        X = df[self.clustering_features].to_numpy()

        # Modify DataFrame inplace for better memory efficiency
        df = df.copy()
        df['cluster'] = self.kmeans.predict(X)
        df['cluster_name'] = df['cluster'].map(self.cluster_names)
        return df

    def get_cluster_means(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cluster means for provided dataframe

        Args:
            df: Dataframe with cluster assignments
        Returns:
            Cluster means with counts
        """
        # More efficient aggregation using a list of columns
        agg_dict = {feature: 'mean' for feature in self.clustering_features}
        agg_dict['cluster'] = 'count'

        cluster_means = (df.groupby(['cluster', 'cluster_name'])
                         .agg(agg_dict)
                         .round(3)
                         .rename(columns={'cluster': "player_count"})
                         .reset_index())
        return cluster_means

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a pickle file

        Args:
            filepath: Path to save the model
        """
        filepath = MODEL_FILE
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, filepath: str) -> 'PokerPlayerClusterer':
        """
        Load the trained model from a pickle file

        Args:
            filepath: Path to the saved model
        Returns:
            Loaded PokerPlayerClusterer instance
        """
        filepath = MODEL_FILE
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            return pickle.load(f)

def main():
    # Create sample data more efficiently
    np.random.seed(42)
    n_samples = 100
    sample_data = {
        feature: np.random.uniform(0, 1, n_samples)
        for feature in ['aggression', 'bluff_frequency', 'fold_rate', 'raise_frequency']
    }
    df = pd.DataFrame(sample_data)

    try:
        # Create and fit the clusterer
        clusterer = PokerPlayerClusterer()
        clusterer.fit(df, clustering_features=list(sample_data.keys()))

        # Test prediction
        results = clusterer.predict(df)

        # Get and display cluster means
        cluster_means = clusterer.get_cluster_means(results)
        print("\nCluster Means:")
        print(cluster_means)

        # Test save and load functionality
        model_path = os.path.join(MODEL_DIR, 'test_model_fakeData.pkl')
        clusterer.save_model(model_path)
        loaded_clusterer = PokerPlayerClusterer.load_model(model_path)
        print("\nModel successfully saved and loaded!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
