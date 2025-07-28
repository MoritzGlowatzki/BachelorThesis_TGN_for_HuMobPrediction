import numpy as np
import pandas as pd

from _1_data_IO import load_csv_file, store_csv_file


# --------- Evaluation Metrics --------- #

def compute_dwt(df):
    """
    Displacement Weighted by Time: Mean Euclidean distance between predicted and true locations.
    """
    distances = np.sqrt((df["x_true"] - df["x_pred"]) ** 2 + (df["y_true"] - df["y_pred"]) ** 2)
    return distances.mean()


def compute_geoblue(df, sigma=5.0):
    """
    GeoBlue score: Higher is better (Gaussian kernel of Euclidean distance).
    Typical sigma = 5 (you can tune based on resolution of grid).
    """
    distances = np.sqrt((df["x_true"] - df["x_pred"]) ** 2 + (df["y_true"] - df["y_pred"]) ** 2)
    return np.mean(np.exp(-distances ** 2 / (2 * sigma ** 2)))


# --------- Main Evaluation Pipeline --------- #

def evaluate(pred_path, true_path):
    pred_df = load_csv_file(pred_path)
    true_df = load_csv_file(true_path)

    # Merge on uid, day, time (assuming these exist)
    merged = pd.merge(
        true_df,
        pred_df,
        on=["uid", "d", "t"],
        suffixes=('_true', '_pred')
    )

    if merged.empty:
        raise ValueError("No matching entries between prediction and ground truth files.")

    store_csv_file("./data/result/cityD_final_comparison.csv", merged)

    # Compute metrics
    dwt_score = compute_dwt(merged[["x_true", "y_true", "x_pred", "y_pred"]])
    geoblue_score = compute_geoblue(merged[["x_true", "y_true", "x_pred", "y_pred"]])

    print(f"DWT (Mean Displacement): {dwt_score:.4f}")
    print(f"GeoBlue Score: {geoblue_score:.4f}")

    return dwt_score, geoblue_score


# --------- Script Entry Point --------- #

if __name__ == "__main__":
    PREDICTION_RESULT_PATH = "./data/result/cityD_prediction_result.csv"
    TRUE_DATA_PATH = "./data/dataset_humob_2024/full_city_data/cityD-dataset.csv"
    evaluate(PREDICTION_RESULT_PATH, TRUE_DATA_PATH)
