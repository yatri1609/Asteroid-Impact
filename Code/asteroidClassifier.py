import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np

# Load the dataset
file_path = '../Data/processed_orbits.csv'
asteroid_data = pd.read_csv(file_path)

# Drop the 'Unnamed: 0' column
asteroid_data = asteroid_data.drop(columns=['Unnamed: 0'])

# Check for and handle missing values (simple strategy: drop rows with missing values)
asteroid_data_cleaned = asteroid_data.dropna()

# Encode the Target Variable
label_encoder = LabelEncoder()
asteroid_data_cleaned['Object.Classification'] = label_encoder.fit_transform(
    asteroid_data_cleaned['Object.Classification'])

# Split the data into features and target variable
X = asteroid_data_cleaned.drop(columns=['Object.Name', 'Object.Classification'])  # Features
y = asteroid_data_cleaned['Object.Classification']  # Target variable

# 3. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the Classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(accuracy, conf_matrix)

# Parameters grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [None, 10, 20],  # Maximum depth of the trees
}

# Grid search for Random Forest
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best parameters and best score
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

print(best_params_rf, best_score_rf)

# Train a Support Vector Machine classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the SVM Classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print(accuracy_svm, conf_matrix_svm)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize both classifiers
rf_classifier_cv = RandomForestClassifier(random_state=42)
svm_classifier_cv = SVC(random_state=42)

# Perform cross-validation and compute mean accuracy
# Random Forest
cv_scores_rf = cross_val_score(rf_classifier_cv, X_train_scaled, y_train, cv=5, n_jobs=-1)

# SVM
cv_scores_svm = cross_val_score(svm_classifier_cv, X_train_scaled, y_train, cv=5, n_jobs=-1)

print(cv_scores_rf.mean(), cv_scores_svm.mean())


# Investigating high accuracy of RF model
# Define function to plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum y-values plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to generate the learning curve.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Plot learning curve for Random Forest Classifier
plot_learning_curve(rf_classifier, "Learning Curve (Random Forest)", X_train, y_train, cv=5, n_jobs=-1)
plt.show()

# Plot learning curve for SVM Classifier
plot_learning_curve(svm_classifier, "Learning Curve (SVM)", X_train_scaled, y_train, cv=5, n_jobs=-1)
plt.show()