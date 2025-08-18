#!/bin/bash

# Install necessary packages (GeoBLEU and DTW metric)
pip install git+https://github.com/yahoojapan/geobleu.git

# Preprocess data
python -u _2a_data_preprocessing.py --city="D" --process="trajectory,user,location" 

# Create TemporalDataset
python -u _2b_dataset.py --city="D" --interpol=True --small=True --feats=True

# Train TGN model
python -u _3_model_training.py --city="D" --epochs=50 --interpol=True --small=True --feats=True --neg_sampling_ratio=50

# Predict with TGN model
python -u _4_predict.py --city="D" --small=True --interpol=True --feats=True --model="last_model_50_on_D_True_True"

# Evaluate predictions
python -u _5_evaluate.py --city="D" --model="last_model_50_on_D_True_True"