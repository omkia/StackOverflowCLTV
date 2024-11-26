import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Ignore any warnings for a cleaner output

# Load your dataset
df = pd.read_csv('rawmodel.csv')

# Define your features (X) and target (y)
X = df.drop(['overalluservalue', 'reputation', 'sumweightTag', 'sumviewcount', 'sumascore', 'oacans'], axis=1)  # Input features
y = df['overalluservalue']  # Target column

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['mse', 'friedman_mse']
}

# Initialize the DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)

# Perform GridSearchCV to find the best parameters
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid,
                              scoring='neg_mean_squared_error', cv=3, verbose=1)

# Fit the model
grid_search_dt.fit(X_train, y_train)

# Get the best parameters
best_dt_params = grid_search_dt.best_params_
print("Best Parameters from GridSearch for Decision Tree:", best_dt_params)

# Train the Decision Tree Regressor with the best parameters
best_dt_model = grid_search_dt.best_estimator_
best_dt_model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred_dt = best_dt_model.predict(X_test)

# Calculate performance metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree - MSE: {mse_dt:.4f}, R-squared: {r2_dt:.4f}")

# Plot the decision tree structure
plt.figure(figsize=(20,10))
plot_tree(best_dt_model, filled=True, feature_names=df.columns[:-6], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Visualize the GridSearchCV results using a heatmap
results = pd.DataFrame(grid_search_dt.cv_results_)
scores_matrix = results.pivot(index='param_min_samples_leaf', columns='param_max_depth', values='mean_test_score')

plt.figure(figsize=(10, 6))
sns.heatmap(scores_matrix, annot=True, cmap="viridis")
plt.title("Grid Search MSE Heatmap")
plt.xlabel("Max Depth")
plt.ylabel("Min Samples Leaf")
plt.show()
# Best Parameters from GridSearch for Decision Tree: {'criterion': 'friedman_mse', 'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
# Decision Tree - MSE: 13970349.2672, R-squared: 0.2640
