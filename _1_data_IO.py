import pandas as pd

def load_csv_file(path):
    return pd.read_csv(path)

def store_csv_file(path, content):
    content.to_csv(path, index=False)


if __name__ == "__main__":
    # read the CSV file
    RAW_DATA_PATH = "./data/dataset_humob_2024/cityD_challengedata.csv"
    data = load_csv_file(RAW_DATA_PATH)

    # filter only rows where uid == 0
    RAW_SMALL_DATA_PATH = "./data/dataset_humob_2024/full_city_data/cityA-dataset-small.csv"
    store_csv_file(RAW_SMALL_DATA_PATH, data[data["uid"] == 0])

    # filter only rows where x and y == 999
    RAW_PREDICT_DATA_PATH = "./data/dataset_humob_2024/cityD_to_predict.csv"
    store_csv_file(RAW_PREDICT_DATA_PATH, data[(data["x"] == 999) & (data["y"] == 999)])
