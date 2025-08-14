#!/bin/bash

# Install necessary packages (GeoBLEU and DTW metric)
# pip install git+https://github.com/yahoojapan/geobleu.git

# Preprocess data
# python -u _2a_data_preprocessing.py --city="C" --process="trajectory,user,location" 

# Create TemporalDataset
python -u _2b_dataset.py --city="D" --interpol=False --small=True

# Train TGN model
python -u _3_model_training.py --city="D" --epochs=10 --interpol=False --small=True