import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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

# 1. Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
kendall_dt, _ = kendalltau(y_test, y_pred_dt)

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
kendall_rf, _ = kendalltau(y_test, y_pred_rf)

# 3. XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
kendall_xgb, _ = kendalltau(y_test, y_pred_xgb)

# 4. Support Vector Regressor (SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
kendall_svr, _ = kendalltau(y_test, y_pred_svr)

# Print comparison of model performances and Kendall's Tau
print("Model Performance and Kendall's Tau Comparison:")
print(f"Decision Tree - MSE: {mse_dt:.4f}, R-squared: {r2_dt:.4f}, Kendall's Tau: {kendall_dt:.4f}")
print(f"Random Forest - MSE: {mse_rf:.4f}, R-squared: {r2_rf:.4f}, Kendall's Tau: {kendall_rf:.4f}")
print(f"XGBoost - MSE: {mse_xgb:.4f}, R-squared: {r2_xgb:.4f}, Kendall's Tau: {kendall_xgb:.4f}")
print(f"SVR - MSE: {mse_svr:.4f}, R-squared: {r2_svr:.4f}, Kendall's Tau: {kendall_svr:.4f}")

# from sklearn.tree import export_graphviz
# import graphviz
#
# # Export the decision tree to a DOT format file
# dot_data = export_graphviz(dt_model, out_file=None,
#                            feature_names=X.columns,
#                            filled=True, rounded=True,
#                            special_characters=True)
#
# # Generate a graph from the DOT data
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree")  # Saves the tree as a file
# graph.view()  # Opens the visualization


import matplotlib.pyplot as plt

# Feature importance plot for Random Forest
importances = rf_model.feature_importances_
plt.barh(X.columns, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.show()

# Decision Tree - MSE: 14444182.5005, R-squared: 0.2391
# Random Forest - MSE: 8017403.3735, R-squared: 0.5776
# XGBoost - MSE: 8643635.7575, R-squared: 0.5446
# SVR - MSE: 18528464.8744, R-squared: 0.0239

