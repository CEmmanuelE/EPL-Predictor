Premier League Match Outcome Prediction
This project predicts the outcome of English Premier League (EPL) matches using machine learning techniques, specifically Naive Bayes, Random Forest, and Logistic Regression models. The project spans several seasons of historical EPL data and evaluates model performance on both a validation set (2023-2024 season) and a test set (2024-2025 season).
Repository Structure
The repository contains three key Python scripts:
	1	data_preparation.py
	2	data_visualisation.py
	3	model_training.py
Each script handles a separate part of the process from preparing data, visualizing features, to training machine learning models.

1. Data Preparation (data_preparation.py)
This script collects and processes data from multiple seasons of EPL matches. Key features such as GoalDifference, HomeTeamForm, and AwayTeamForm are created to enhance model performance.
Steps Performed:
	•	Loads all Excel files from a specified folder containing multiple EPL seasons.
	•	Standardizes column formats and concatenates the datasets.
	•	Creates new features: GoalDifference, HomeTeamPoints, AwayTeamPoints, and rolling average TeamForm for both home and away teams.
	•	Saves the cleaned dataset as combined_seasons_data_prepared.csv.
Key Files:
	•	Input: Excel files from various seasons.
	•	Output: combined_seasons_data_prepared.csv

2. Data Visualization (data_visualisation.py)
This script visualizes key aspects of the prepared data to better understand relationships between variables such as GoalDifference and HomeTeamForm, and their impact on match outcomes. The visualizations also compare the performance of different models based on accuracy and time.
Visualizations:
	•	Box plots for GoalDifference and HomeTeamForm by match outcome.
	•	Correlation matrix of main features.
	•	Chi-square tests for the significance of features like HomeTeamForm.
	•	Bar charts comparing the accuracy of different machine learning models.
	•	Line plots to show the trade-off between feature complexity and computational time for Naive Bayes and Random Forest.
Key Files:
	•	Input: combined_seasons_data_prepared.csv
	•	Output: Visualizations generated through matplotlib and seaborn.

3. Model Training (model_training.py)
This script handles the model training and evaluation process. Three machine learning models—Logistic Regression, Random Forest, and Naive Bayes—are trained on historical EPL data. Grid search is used to fine-tune hyperparameters for Random Forest.
Steps Performed:
	1	Training Models:
	◦	Uses cross-validation and hyperparameter tuning for Random Forest via GridSearchCV.
	◦	Models are trained on features such as HomeTeamForm, AwayTeamForm, and GoalDifference.
	2	Evaluation on Validation Set:
	◦	The 2023-2024 season is used to validate model performance. Key metrics such as accuracy, precision, recall, and confusion matrices are calculated.
	◦	Model performance is visualized, showing trade-offs between accuracy and computational cost.
	3	Testing on 2024-2025 Season:
	◦	Predictions are made on the upcoming season (2024-2025). These predictions are saved to a CSV file for further evaluation as more matches are played.
Key Files:
	•	Input: combined_seasons_data_prepared.csv (training data) and all-euro-data-2023-2024.xlsx (validation data).
	•	Output: predictions_2024_2025_season.csv (predicted outcomes for the upcoming season).

Requirements
To run this project, ensure the following Python libraries are installed:
“pip install pandas matplotlib seaborn scikit-learn openpyxl”

Usage
1. Data Preparation
“python data_preparation.py”
This script processes the raw Excel files, generates new features, and saves a cleaned dataset.
2. Data Visualization
“python data_visualisation.py”
This script generates several visualizations that help understand key features and evaluate model performance.
3. Model Training and Prediction
“python model_training.py”
This script trains the machine learning models, validates them using the 2023-2024 season data, and predicts outcomes for the 2024-2025 season.

Output
Key Results:
	•	The Random Forest model achieved an accuracy of 100% on the validation set (2023-2024).
	•	Naive Bayes and Logistic Regression showed slightly lower accuracies but remained competitive.
	•	Predictions for the 2024-2025 season are saved as predictions_2024_2025_season.csv.

Future Work
Future work can involve:
	•	Incorporating additional features such as player injuries, weather conditions, or managerial changes.
	•	Expanding the model to predict outcomes for multiple football leagues.
	•	Continuous evaluation as the 2024-2025 season progresses, allowing for real-time model adjustments.
