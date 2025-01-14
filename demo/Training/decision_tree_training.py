import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Load the dataset
file_path = "training_dataset.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")

# Drop rows with missing target variable
data = data.dropna(subset=["rating"])

# Drop rows with any missing values in the dataset
data = data.dropna()

# Encode the target variable
label_encoder = LabelEncoder()
data["rating"] = label_encoder.fit_transform(data["rating"])

# Select features and target
X = data.drop(columns=["Pattern.Name", "Hole.id", "rating"])
y = data["rating"]

# Split data into training and testing sets
# Stratified split for better class representation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# RandomForest GridSearchCV
rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(
    rf, param_grid=rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
rf_grid_search.fit(X_train, y_train)

print("Best Parameters for RandomForest:", rf_grid_search.best_params_)
print("Best Accuracy for RandomForest:", rf_grid_search.best_score_)

# DecisionTree GridSearchCV
# dt_param_grid = {
#     "max_depth": [3, 5, 10, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }

# dt = DecisionTreeClassifier(random_state=42)
# dt_grid_search = GridSearchCV(
#     dt, param_grid=dt_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
# )
# dt_grid_search.fit(X_train, y_train)

# print("Best Parameters for DecisionTree:", dt_grid_search.best_params_)
# print("Best Accuracy for DecisionTree:", dt_grid_search.best_score_)


# Extract the best parameters from GridSearchCV
best_params = rf_grid_search.best_params_

# Initialize the RandomForestClassifier with the best parameters
rf_best = RandomForestClassifier(**best_params, random_state=42)

# Train the model on the entire dataset (X, y)
rf_best.fit(X, y)

print("Number of features the model expects:", rf_best.n_features_in_)
# Output to confirm training
print("RandomForestClassifier trained with best parameters on the entire dataset.")

import pickle

# Save the trained model to a file
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(rf_best, model_file)

print("Model saved as random_forest_model.pkl")

# To load the model later
with open("random_forest_model.pkl", "rb") as model_file:
    rf_loaded = pickle.load(model_file)
print("Model loaded successfully.")
