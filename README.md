# Temporal Graph Network (TGN) for Human Mobility Prediction

<div align="center">
 
> <img src="https://upload.wikimedia.org/wikipedia/commons/c/c8/Logo_of_the_Technical_University_of_Munich.svg" width="120"/>
>
> #### School of Computation, Information and Technology – Informatics<br> Technische Universität München (TUM)  
>
> _Bachelor's Thesis in Information Systems (Wirtschaftsinformatik)_
>
> ### "Human Mobility Prediction with Continuous-Time Dynamic Graphs and Temporal Graph Networks" <br> *(Vorhersage menschlicher Mobilität mit kontinuierlich-zeitdynamischen Graphen und Temporal Graph Networks)*  
>
> **Author:** Moritz Glowatzki  
> **Examiner:** Prof. Dr. Martin Werner  
> **Supervisor:** MSc. Paul Walther  

</div>

<br>

## 📂 Project Structure

```
.
├── 📂 data/                                 # Data folder [Completely excluded from Git]
│   ├── 📂 dataset_humob_2024/                   # Raw HUMOB 2024 data
│   │   ├── 📂 full_city_data/                       # Full dataset including ground truth for challenge evaluation
│   │   ├── 📊 city*_challengedata.csv               # Challenge data files for every city (A, B, C, D)
│   │   ├── 📊 POI_datacategories.csv                # Points-of-interest data categories
│   │   └── 📊 POIdata_city*.csv                     # POI data files for every city (A, B, C, D)
│   ├── 🗂️ processed/                               # Processed data in .pt format
│   ├── 📂 raw/                                  # Data after running '_2a_data_preprocessing.py'
│   │   ├── 📊 city*_location_features.csv           # Location features for every city (A, B, C, D)
│   │   ├── 📊 city*_prediction_data.csv             # Raw data that needs to be predicted later (x=999 and y=999)
│   │   ├── 📊 city*_trajectory_data.csv             # Edge features for every city (A, B, C, D)
│   │   └── 📊 city*_user_features.csv               # User features for every city (A, B, C, D)
│   └── 📂 result/                               # Final predictions
├── 📂️ model_training_runs/                  # Training logs and checkpoints
├── 📂 output/                               # Visualization plots
├── 📄 .gitignore
├── 🐍 1_data_IO.py                          # Data input/output
├── 🐍 _2a_data_preprocessing.py             # Data preprocessing pipeline
├── 🐍 _2b_dataset.py                        # UserLocationInteractionDataset generator
├── 🐍 _3_model_training.py                  # Model training
├── 🐍 _4_predict.py                         # Model inference
├── 🐍 _5_evaluate.py                        # Evaluation
├── 📜 LICENSE
├── 📘 README.md
├── ⚙️ requirements.txt
├── 🐍 visualize_data.py                     # Custom data visualization
└── 🐍 visualize_trajectory.py               # Custom Trajectory visualization
```

<br>

## 📊 Dataset Information

This project uses the **LYMob-4Cities: Multi-City Human Mobility Dataset** [1] from the **HuMob Challenge 2024**.
It is publicly accessible via [Zenodo](https://zenodo.org/records/14219563).

<ins>*Disclaimer*</ins>: The dataset is subject to the terms and conditions provided by the HuMob Challenge organizers.
Please refer to their official documentation for details on licensing and citation requirements.

[1] Yabe, T., Tsubouchi, K., Shimizu, T., Sekimoto, Y., Sezaki, K., Moro, E., & Pentland, A. (2024). YJMob100K:
City-scale and longitudinal dataset of anonymized human mobility trajectories. Scientific Data, 11(1),
397. https://www.nature.com/articles/s41597-024-03237-9

<br>

## 📜 License

This project is licensed under the MIT License.
See the [LICENSE](/LICENSE) file for details.
 
