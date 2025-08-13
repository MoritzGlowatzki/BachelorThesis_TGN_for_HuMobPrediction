import numpy as np
import pandas as pd
from geobleu import calc_dtw_orig, calc_geobleu_orig

from _1_data_IO import load_csv_file, store_csv_file


# --------- Evaluation Metrics --------- #

def compute_dtw(df):
    """
    Compute the Dynamic Time Warping (DTW) score between true and predicted trajectories (lower is better).

    DTW measures the similarity between two temporal sequences which may vary in speed.
    Lower DTW values indicate better alignment (less temporal/spatial discrepancy).
    """
    # distances = np.sqrt((df["x_true"] - df["x_pred"]) ** 2 + (df["y_true"] - df["y_pred"]) ** 2)
    # return distances.mean()

    true = df[["x_true", "y_true"]].values
    pred = df[["x_pred", "y_pred"]].values
    return calc_dtw_orig(true, pred)


def compute_geobleu(df):
    """
    Compute the GeoBLEU score between true and predicted locations (higher is better).

    GeoBLEU applies a Gaussian kernel to the spatial distance between predicted and true points,
    rewarding predictions that are spatially close.
    """
    # distances = np.sqrt((df["x_true"] - df["x_pred"]) ** 2 + (df["y_true"] - df["y_pred"]) ** 2)
    # return np.mean(np.exp(-distances ** 2 / (2 * sigma ** 2)))

    true = df[["x_true", "y_true"]].values
    pred = df[["x_pred", "y_pred"]].values
    return calc_geobleu_orig(true, pred, max_n=5, beta=0.5)


def compute_mean_euclidean_dist(df):
    """
    Compute the mean Euclidean distance between true and predicted points.

    This is a straightforward pointwise error metric that does not account for temporal distortions or sequence alignment.
    It measures the average spatial discrepancy between matched points.
    """
    distances = np.sqrt((df["x_true"] - df["x_pred"]) ** 2 + (df["y_true"] - df["y_pred"]) ** 2)
    return distances.mean()


# --------- Main Evaluation Pipeline --------- #
def evaluate(pred_path, true_path):
    pred_df = load_csv_file(pred_path)
    true_df = load_csv_file(true_path)

    print("Merge predictions and ground truth into one dataframe...")

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

    print("Start computing metric scores...")

    # Compute metrics
    dwt_score = compute_dtw(merged)
    geobleu_score = compute_geobleu(merged)
    # mean_dist_val = compute_mean_euclidean_dist(merged)

    print(f"DTW Score: {dwt_score:.4f}")
    print(f"GeoBLEU Score: {geobleu_score:.4f}")
    # print("Mean Euclidean Distance:", mean_dist_val)

    return dwt_score, geobleu_score


if __name__ == "__main__":
    PREDICTION_RESULT_PATH = "./data/result/cityD_prediction_result.csv"
    TRUE_DATA_PATH = "./data/dataset_humob_2024/full_city_data/cityD-dataset.csv"
    evaluate(PREDICTION_RESULT_PATH, TRUE_DATA_PATH)
