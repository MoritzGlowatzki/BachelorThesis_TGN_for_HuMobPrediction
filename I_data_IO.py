import pandas as pd

RAW_DATA_PATH = "./data/cityA-dataset.csv"
RAW_SMALL_DATA_PATH = "./data/cityA-dataset-small.csv"

def load_csv_file(path):
    return pd.read_csv(path)

def store_csv_file(path, content):
    content.to_csv(path, index=False)


if __name__ == "__main__":
    # read the CSV file
    data = load_csv_file(RAW_DATA_PATH)

    # Filter only rows where uid == 0
    store_csv_file(RAW_SMALL_DATA_PATH, data[data["uid"] <= 1])
