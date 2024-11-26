import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

# Initialize the RandomForestRegressor with the best parameters
best_rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=4,
    min_samples_split=2,
    bootstrap=True,
    random_state=42
)

# Train the model
best_rf_model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred_rf = best_rf_model.predict(X_test)

# Calculate performance metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Calculate Kendall's Tau
kendall_rf, _ = kendalltau(y_test, y_pred_rf)

# Display the results
print(f"Random Forest with Best Parameters - MSE: {mse_rf:.4f}, R-squared: {r2_rf:.4f}, Kendall's Tau: {kendall_rf:.4f}")
# Random Forest with Best Parameters - MSE: 13323458.1854, R-squared: 0.2981, Kendall's Tau: 0.4466
