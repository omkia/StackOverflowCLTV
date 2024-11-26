import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import time
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

# Define the parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with parallel processing
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                              scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

# Perform grid search
start_time = time.time()
grid_search_rf.fit(X_train, y_train)
print(f"Grid Search Execution Time: {time.time() - start_time:.2f} seconds")

# Best parameters found by GridSearchCV
print("Best Parameters from GridSearch for Random Forest:", grid_search_rf.best_params_)

# Train and predict using the best parameters
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Calculate performance metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Display the results
print(f"Random Forest Performance with Best Parameters:")
print(f"MSE: {mse_rf:.4f}")
print(f"R-squared: {r2_rf:.4f}")
# Grid Search Execution Time: 6249.45 seconds
# Best Parameters from GridSearch for Random Forest: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
# Random Forest Performance with Best Parameters:
# MSE: 13323458.1854
# R-squared: 0.2981
