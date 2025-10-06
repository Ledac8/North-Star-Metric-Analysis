import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
DATA_FILE = 'customer_segmentation_data.csv'
N_CLUSTERS = 3 # We expect 3 main personas: e-commerce, agency, dev

# Features to use for clustering (excluding user_id)
FEATURE_COLUMNS = [
    'images_generated_monthly',
    'api_calls_monthly',
    'login_frequency_weekly',
    'templates_used_monthly',
    'custom_generation_minutes_monthly',
    'feature_x_used'
]

# --- Main Clustering Analysis ---
def perform_customer_segmentation(data_path, feature_cols, n_clusters):
    """
    Loads data, preprocesses it, performs K-Means clustering,
    and returns labeled clusters with centroids.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure the CSV is in the same directory.")
        return None, None, None

    # Select features for clustering
    X = df[feature_cols]

    # Handle potential missing values (e.g., fill with mean or median)
    X = X.fillna(X.mean())

    # Standardize the features - crucial for K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled_df)

    # Calculate cluster centroids (mean of features for each cluster)
    cluster_centroids = df.groupby('cluster')[feature_cols].mean()
    # Also get the original (unscaled) centroids for easier interpretation
    original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    original_centroids_df = pd.DataFrame(original_centroids, columns=feature_cols)
    original_centroids_df.index.name = 'cluster'


    return df, cluster_centroids, original_centroids_df

def label_clusters(centroids_df):
    """
    Attempts to label clusters based on the simulated persona characteristics.
    This is an interpretative step based on domain knowledge.
    """
    labels = {}
    for i, row in centroids_df.iterrows():
        if row['api_calls_monthly'] > 200 and row['login_frequency_weekly'] < 3:
            labels[i] = "Developers (API-Centric)"
        elif row['images_generated_monthly'] > 100 and row['templates_used_monthly'] > 15:
            labels[i] = "E-commerce Business Owners"
        elif row['images_generated_monthly'] > 50 and row['custom_generation_minutes_monthly'] > 30:
            labels[i] = "Marketing Agencies"
        else:
            labels[i] = f"Unidentified Cluster {i}" # Fallback

    return labels

def plot_clusters(df, x_feature, y_feature, cluster_col, labels_map=None, plot_file='clusters_plot.png'):
    """
    Generates and saves a 2D scatter plot of the clusters.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x=x_feature,
        y=y_feature,
        hue=cluster_col,
        palette='viridis',
        s=100,
        alpha=0.7
    )

    if labels_map:
        handles, _labels = plt.gca().get_legend_handles_labels()
        new_labels = [labels_map.get(int(lbl), f"Cluster {lbl}") for lbl in _labels]
        plt.legend(handles=handles, labels=new_labels, title="User Clusters")

    plt.title(f'Customer Segments based on {x_feature} vs. {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_file)
    plt.close()
    print(f"Cluster plot saved to {plot_file}")

# --- Execute Analysis ---
if __name__ == "__main__":
    print(f"Running customer segmentation with {N_CLUSTERS} clusters...")
    clustered_df, scaled_centroids, original_centroids = perform_customer_segmentation(
        DATA_FILE, FEATURE_COLUMNS, N_CLUSTERS
    )

    if clustered_df is not None:
        print("\n--- Cluster Centroids (Mean Feature Values for Each Cluster) ---")
        print("These values represent the 'average' user for each cluster, on the original scale.")
        print(original_centroids.to_string())

        # Label the clusters
        cluster_labels = label_clusters(original_centroids)
        print("\n--- Interpreted Cluster Labels ---")
        for cluster_id, label in cluster_labels.items():
            print(f"Cluster {cluster_id}: {label}")

        # Add interpreted labels to the DataFrame for plotting legend
        clustered_df['cluster_label'] = clustered_df['cluster'].map(cluster_labels)

        print("\n--- Proposing Engagement Strategies for Each Cluster ---")
        for cluster_id in sorted(cluster_labels.keys()):
            label = cluster_labels[cluster_id]
            print(f"\nCluster {cluster_id}: **{label}**")
            # Example strategies based on expected persona
            if "Developers" in label:
                print("- Strategy: Focus on API documentation, SDKs, webhook integrations, and developer community support. Promote new API features.")
            elif "E-commerce" in label:
                print("- Strategy: Emphasize bulk image generation, product photo editing tools, brand kit consistency, and e-commerce platform integrations. Offer templates for product listings.")
            elif "Marketing Agencies" in label:
                print("- Strategy: Highlight team collaboration features, client management tools, diverse creative templates, and advanced customization options. Showcase portfolio-building features.")
            else:
                print("- Strategy: Further investigation needed to define this cluster's specific needs and propose tailored engagement.")

        # Generate and save the 2D plot
        # Choose two features that you expect to show good separation for visualization
        # 'images_generated_monthly' vs 'api_calls_monthly' is a good starting point for these personas
        plot_clusters(clustered_df, 'images_generated_monthly', 'api_calls_monthly', 'cluster', cluster_labels, 'customer_clusters.png')
        print("\nAnalysis complete. Check 'customer_clusters.png' for the visualization.")

# Optional: To run this script locally, ensure you have these libraries installed:
# pip install pandas scikit-learn matplotlib seaborn