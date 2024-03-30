import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.model_selection
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as pyplt
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
file_path = '../Data/processed_impacts.csv'
impact_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(impact_data.head())

# Drop the 'Unnamed: 0' column, X and Object.Name as it is unique for each object
impact_data_cleaned = impact_data.drop(columns=['Unnamed: 0', 'Object.Designation..', 'X'])

# Check for missing values
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

# Plot correlation heatmap for the impact data
colormap = pyplt.cm.viridis
sns.heatmap(impact_data_cleaned.corr(), cmap=colormap, annot=True)
plt.show()

# Drop Period Start and Period End due to multi-collinearity as Period is already in the data
impact_data_cleaned = impact_data_cleaned.drop(columns=['Period Start', 'Period End'])
# Drop Torino Scale and Palermo scale max due to multi-collinearity as well
impact_data_cleaned = impact_data_cleaned.drop(columns=['Torino.Scale..max..', 'Palermo.Scale..max..'])

# Define the features (X) and the target (y)
X = impact_data_cleaned.drop(columns=['Potential.Impacts..'])
y = impact_data_cleaned['Potential.Impacts..']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to test
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Stochastic Gradient Descent": SGDRegressor(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(random_state=42))
}

# Initialize dictionary to store results
results = {}

# Train and evaluate each model without any feature engineering
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on the testing set
    y_pred = model.predict(X_test)
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Store results
    results[name] = {'MSE': mse, 'R2': r2}

print("Results of models without any feature engineered data:", results)
noFEresults = pd.DataFrame.from_dict(results, orient='index')
print(noFEresults)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Scale the features
scaler = StandardScaler()

# Transform the features
X_poly = poly.fit_transform(X)
X_poly_scaled = scaler.fit_transform(X_poly)

# Split the enhanced features into training and testing sets
X_train_enhanced, X_test_enhanced, y_train, y_test = sklearn.model_selection.train_test_split(X_poly_scaled, y,
                                                                                              test_size=0.2,
                                                                                              random_state=42)

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

print("Results of Poly Scaled data:", results)
polyScaleResults = pd.DataFrame.from_dict(results, orient='index')
print(polyScaleResults)

# Applying Reverse Feature Selection to remove the least important features
# Initialize the model and RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)  # Selecting the top 6 features
# Fit RFE
rfe.fit(X_train, y_train)

# Assuming X_train and X_test are pandas DataFrames
selected_features = X_train.columns[rfe.support_]
print(selected_features)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_selected, y_train)
    # Predict on the testing set
    y_pred = model.predict(X_test_selected)
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Store results
    results[name] = {'MSE': mse, 'R2': r2}

print("Results of RFE selected data:", results)
RFEresults = pd.DataFrame.from_dict(results, orient='index')
print(RFEresults)

# Fine tune KNN hyper-parameters based on the RFE selected data as that model had the highest accuracy
param_grid = {'n_neighbors': range(1, 10)}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_
# Predict on the testing set
y_pred = best_model.predict(X_test_selected)
# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print results
print('Results of fine-tuned KNN model : MSE:', mse, 'R2:', r2)


# Transform the features
X_scaled = scaler.fit_transform(X)

# Split the enhanced features into training and testing sets
X_train_enhanced, X_test_enhanced, y_train, y_test = sklearn.model_selection.train_test_split(X_scaled, y,
                                                                                              test_size=0.2,
                                                                                              random_state=42)

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

print("Results of Scaled data:", results)
scaledResults = pd.DataFrame.from_dict(results, orient='index')
print(scaledResults)

# Based on the analysis scaled data performs the best when it comes to predicting Potential Impacts
# Let's try the models with just Period, Impact Probability, Diameter and Velocity

X_new = X[['Period','Impact.Probability..cumulative.','Vinfinity..km.s.','H..mag.']]
X_new_scaled = scaler.fit_transform(X_new)

# Split the enhanced features into training and testing sets
X_train_enhanced, X_test_enhanced, y_train, y_test = sklearn.model_selection.train_test_split(X_new_scaled, y,
                                                                                              test_size=0.2,
                                                                                              random_state=42)


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

print("Results of Chosen Scaled data:", results)
chosenScaledResults = pd.DataFrame.from_dict(results, orient='index')
print(chosenScaledResults)


# Bases on all the results, Random forest seem to perform the best along with the scaled data.
