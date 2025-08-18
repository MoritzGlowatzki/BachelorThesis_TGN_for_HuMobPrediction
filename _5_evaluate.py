import argparse
import numpy as np
import pandas as pd
from geobleu.seq_eval import calc_geobleu_bulk, calc_dtw_bulk
from tqdm import tqdm

from _1_data_IO import load_csv_file, store_csv_file


# --------- Evaluation Metrics --------- #
def compute_dtw(df):
    """
    Compute the Dynamic Time Warping (DTW) score between true and predicted trajectories (lower is better).

    DTW measures the similarity between two temporal sequences which may vary in speed.
    Lower DTW values indicate better alignment (less temporal/spatial discrepancy).
    """
    true = df[["uid", "d", "t", "x_true", "y_true"]].values
    pred = df[["uid", "d", "t", "x_pred", "y_pred"]].values
    return calc_dtw_bulk(true, pred, processes=4)


def compute_geobleu(df):
    """
    Compute the GeoBLEU score between true and predicted locations (higher is better).

    GeoBLEU applies a Gaussian kernel to the spatial distance between predicted and true points,
    rewarding predictions that are spatially close.
    """
    true = df[["uid", "d", "t", "x_true", "y_true"]].values
    pred = df[["uid", "d", "t", "x_pred", "y_pred"]].values
    return calc_geobleu_bulk(true, pred, processes=4)


def compute_mean_euclidean_dist(df):
    """
    Compute the mean Euclidean distance between true and predicted points.
    """
    distances = np.sqrt((df["x_true"] - df["x_pred"]) ** 2 + (df["y_true"] - df["y_pred"]) ** 2)
    return distances.mean()


# --------- Main Evaluation Pipeline --------- #
def evaluate_per_user(pred_path, true_path):
    pred_df = load_csv_file(pred_path)
    true_df = load_csv_file(true_path)

    print("Merge predictions and ground truth into one dataframe...")

    # Merge on uid, day, time
    merged = pd.merge(
        true_df,
        pred_df,
        on=["uid", "d", "t"],
        suffixes=('_true', '_pred')
    )

    if merged.empty:
        raise ValueError("No matching entries between prediction and ground truth files.")
    store_csv_file("./data/result/cityD_final_comparison.csv", merged)

    print("Compute metrics per user and aggregate...")
    
    dtw_score = compute_dtw(merged)
    print(f"DTW score: {dtw_score:.4f}")
    geobleu_score = compute_geobleu(merged)
    print(f"GeoBLEU score: {geobleu_score:.4f}")
    mean_euclidean_dist = compute_mean_euclidean_dist(merged)
    print(f"Mean Euclidean Distance: {mean_euclidean_dist:.4f}")

    return dtw_score, geobleu_score, mean_euclidean_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TGN model predictions against ground truth.")
    parser.add_argument("--city", type=str, default="D", help="City index (e.g., A, B, C, D)")
    parser.add_argument("--model", type=str, default="last_model_D_50", help="Used model")
    args = parser.parse_args()
    
    PREDICTION_RESULT_PATH = f"./data/result/city{args.city}_prediction_result_{args.model}.csv"
    TRUE_DATA_PATH = f"./data/dataset_humob_2024/full_city_data/city{args.city}-dataset.csv"
    evaluate_per_user(PREDICTION_RESULT_PATH, TRUE_DATA_PATH)
