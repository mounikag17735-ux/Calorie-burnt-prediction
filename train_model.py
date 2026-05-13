import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("calories.csv")

# Data preprocessing
df = df.drop('User_ID', axis=1)

df['Gender'] = df['Gender'].map({
    'male': 0,
    'female': 1
})

# Features and target
X = df.drop('Calories', axis=1)

y = df['Calories']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, rf_pred)

r2 = r2_score(y_test, rf_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Save model
pickle.dump(rf_model, open('model.pkl', 'wb'))

print("Model saved successfully")