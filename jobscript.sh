#!/bin/bash

# Install necessary packages (GeoBLEU and DTW metric)
# pip install git+https://github.com/yahoojapan/geobleu.git

# Preprocess data
# python -u _2a_data_preprocessing.py --city="D" --process="trajectory,user,location" 

# Create TemporalDataset
python -u _2b_dataset.py --city="D"

# Train TGN model
python -u _3_model_training.py --city="D" --epochs=50 --save_path="./model_training_runs/best_model_D_50.pt"