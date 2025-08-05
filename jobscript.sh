#!/bin/bash

# Preprocess data
# python -u ./_2a_data_preprocessing.py --city="D" --process="trajectory,user,location" 

# Create TemporalDataset
# python -u ./_2b_dataset.py --city="D"

# Train TGN model
python -u ./_3_model_training.py --city="D" --epochs=50