import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import kendalltau
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

# Define the parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3, verbose=1)
# Best Parameters from GridSearch: {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 500, 'subsample': 0.8}
# MSE: 13773603.6144
# R-squared: 0.2744
# Kendall's Tau: 0.4342

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters from GridSearch:", grid_search.best_params_)

# Train the model using the best parameters
best_xgb_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred_xgb = best_xgb_model.predict(X_test)

# Calculate performance metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
kendall_xgb, _ = kendalltau(y_test, y_pred_xgb)

# Display the results
print(f"XGBoost Performance with Best Parameters:")
print(f"MSE: {mse_xgb:.4f}")
print(f"R-squared: {r2_xgb:.4f}")
print(f"Kendall's Tau: {kendall_xgb:.4f}")
