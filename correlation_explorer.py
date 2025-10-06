import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

# --- Configuration ---
DATA_FILE = 'simulated_user_data.csv'
NSM_TARGET_COLUMN = 'nsm_achieved_week2' # Your defined NSM target
RETENTION_TARGET_COLUMN = 'retention_status_week_8' # For bonus retention analysis

# Columns that represent behavioral features you want to correlate with the NSM
FEATURE_COLUMNS = [
    'image_downloads_week1',
    'smart_crop_used_week1',
    'avg_session_time_min_week1',
    'num_searches_week1',
    'saved_to_favorites_week1'
]

# --- Main Analysis ---
def analyze_nsm_correlations(data_path, nsm_col, feature_cols):
    """
    Loads data, calculates Pearson and Spearman correlations between
    behavioral features and the North Star Metric.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure the CSV is in the same directory.")
        return

    results = []

    for feature in feature_cols:
        if feature in df.columns and nsm_col in df.columns:
            # Drop rows with NaN in either feature or NSM for accurate correlation
            temp_df = df[[feature, nsm_col]].dropna()

            if not temp_df.empty and len(temp_df) > 1: # Need at least 2 data points
                # Pearson Correlation (for linear relationships)
                pearson_corr, pearson_p = pearsonr(temp_df[feature], temp_df[nsm_col])
                # Spearman Correlation (for monotonic relationships, good for ranks/non-normal data)
                spearman_corr, spearman_p = spearmanr(temp_df[feature], temp_df[nsm_col])

                results.append({
                    'Feature': feature,
                    'Pearson_Correlation_with_NSM': pearson_corr,
                    'Pearson_P_Value': pearson_p,
                    'Spearman_Correlation_with_NSM': spearman_corr,
                    'Spearman_P_Value': spearman_p
                })
            else:
                print(f"Warning: Not enough data for correlation between {feature} and {nsm_col}")
        else:
            print(f"Error: Feature '{feature}' or NSM column '{nsm_col}' not found in data.")

    return pd.DataFrame(results)

def analyze_retention_lift(df, nsm_col, retention_col):
    """
    Calculates retention rates for users who did and did not achieve the NSM,
    and the percentage lift.
    """
    nsm_achievers = df[df[nsm_col] == 1]
    non_nsm_achievers = df[df[nsm_col] == 0]

    retention_nsm = nsm_achievers[retention_col].mean() * 100
    retention_non_nsm = non_nsm_achievers[retention_col].mean() * 100

    if retention_non_nsm > 0:
        retention_lift = ((retention_nsm - retention_non_nsm) / retention_non_nsm) * 100
    else:
        retention_lift = float('inf') # Avoid division by zero if no non-achievers retained

    print(f"\n--- Retention Analysis (NSM Achievers vs. Non-Achievers) ---")
    print(f"Retention Rate for NSM Achievers: {retention_nsm:.2f}%")
    print(f"Retention Rate for Non-NSM Achievers: {retention_non_nsm:.2f}%")
    print(f"Retention Lift for NSM Achievers: {retention_lift:.2f}%\n")


# --- Execute Analysis ---
if __name__ == "__main__":
    print(f"Running correlation analysis for NSM: '{NSM_TARGET_COLUMN}'")
    correlation_df = analyze_nsm_correlations(DATA_FILE, NSM_TARGET_COLUMN, FEATURE_COLUMNS)

    if correlation_df is not None:
        print("\n--- Correlation Results with NSM ---")
        print(correlation_df.sort_values(by='Pearson_Correlation_with_NSM', ascending=False).to_string())

    # Optional: Run retention analysis (ensure both columns exist in the DataFrame)
    try:
        df_full = pd.read_csv(DATA_FILE)
        if NSM_TARGET_COLUMN in df_full.columns and RETENTION_TARGET_COLUMN in df_full.columns:
            analyze_retention_lift(df_full, NSM_TARGET_COLUMN, RETENTION_TARGET_COLUMN)
        else:
            print(f"Cannot perform retention analysis: '{NSM_TARGET_COLUMN}' or '{RETENTION_TARGET_COLUMN}' not found in data.")
    except FileNotFoundError:
        pass # Already handled by correlation function