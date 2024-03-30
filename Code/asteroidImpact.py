import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the dataset
file_path = '../Data/processed_impacts.csv'
impact_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(impact_data.head())

# Drop the 'Unnamed: 0' column and Object.Name as it is unique for each object
impact_data_cleaned = impact_data.drop(columns=['Unnamed: 0', 'Object.Designation..', 'X'])

# Step 3: Check for missing values
missing_values = impact_data_cleaned.isnull().sum()

print(missing_values)

# Impute missing values in 'Torino.Scale..max..' with the column's median
median_value = impact_data_cleaned['Torino.Scale..max..'].median()
impact_data_cleaned['Torino.Scale..max..'].fillna(median_value, inplace=True)

# Impute missing values in 'Vinfinity..km.s.' with the column's median
median_value = impact_data_cleaned['Vinfinity..km.s.'].median()
impact_data_cleaned['Vinfinity..km.s.'].fillna(median_value, inplace=True)

# Verify imputation
missing_values_after = impact_data_cleaned.isnull().sum()

print(missing_values_after)

# Define the features (X) and the target (y)
X = impact_data_cleaned.drop(columns=['Potential.Impacts..'])
y = impact_data_cleaned['Potential.Impacts..']

# Plot correlation heatmap for the impact data
sns.heatmap(impact_data_cleaned.corr(), cmap="YlGnBu", annot=True)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize the Linear Regression model
lr_model = sklearn.linear_model.LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict on the testing set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
r2 = sklearn.metrics.r2_score(y_test, y_pred)

print("MSE:", mse, "R2:", r2)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Scale the features
scaler = StandardScaler()

# Transform the features
X_poly = poly.fit_transform(X)
X_poly_scaled = scaler.fit_transform(X_poly)

# Split the enhanced features into training and testing sets
X_train_enhanced, X_test_enhanced, y_train, y_test = sklearn.model_selection.train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)

# Define models to test
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(random_state=42))
}

# Initialize dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_enhanced, y_train)
    # Predict on the testing set
    y_pred = model.predict(X_test_enhanced)
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Store results
    results[name] = {'MSE': mse, 'R2': r2}

print(results)

# Initialize and evaluate the KNN model again to ensure stability
knn_model = KNeighborsRegressor()
knn_model.fit(X_train_enhanced, y_train)
y_pred_knn = knn_model.predict(X_test_enhanced)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(mse_knn, r2_knn)
