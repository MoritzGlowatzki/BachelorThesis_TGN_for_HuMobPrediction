import pandas as pd
from geobleu import calc_dtw_orig, calc_geobleu_orig
from tqdm import tqdm

from _1_data_IO import load_csv_file, store_csv_file


# --------- Evaluation Metrics --------- #

def compute_dtw(df):
    """
    Compute the Dynamic Time Warping (DTW) score between true and predicted trajectories (lower is better).

    DTW measures the similarity between two temporal sequences which may vary in speed.
    Lower DTW values indicate better alignment (less temporal/spatial discrepancy).
    """
    true = df[["x_true", "y_true"]].values
    pred = df[["x_pred", "y_pred"]].values
    return calc_dtw_orig(true, pred)


def compute_geobleu(df):
    """
    Compute the GeoBLEU score between true and predicted locations (higher is better).

    GeoBLEU applies a Gaussian kernel to the spatial distance between predicted and true points,
    rewarding predictions that are spatially close.
    """
    true = df[["x_true", "y_true"]].values
    pred = df[["x_pred", "y_pred"]].values
    return calc_geobleu_orig(true, pred, max_n=5, beta=0.5)


# --------- Main Evaluation Pipeline (Per User) --------- #
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

    results = []
    for uid in tqdm(merged["uid"].unique(), desc="Evaluating users"):
        group = merged[merged["uid"] == uid]
        dwt_score = compute_dtw(group)
        geobleu_score = compute_geobleu(group)

        print(f"DTW Score for User {uid}: {dwt_score:.4f}")
        print(f"GeoBLEU Score for User {uid}: {geobleu_score:.4f}")

        results.append({
            "uid": uid,
            "DTW": dwt_score,
            "GeoBLEU": geobleu_score,
        })

    per_user_metrics = pd.DataFrame(results)

    # Compute overall averages
    overall_metrics = per_user_metrics[["DTW", "GeoBLEU"]].mean()

    print(f"\nOverall DTW Score: {overall_metrics['DTW']:.4f}")
    print(f"Overall GeoBLEU Score: {overall_metrics['GeoBLEU']:.4f}")

    return overall_metrics


if __name__ == "__main__":
    PREDICTION_RESULT_PATH = "./data/result/cityD_prediction_result.csv"
    TRUE_DATA_PATH = "./data/dataset_humob_2024/full_city_data/cityD-dataset.csv"
    evaluate_per_user(PREDICTION_RESULT_PATH, TRUE_DATA_PATH)
