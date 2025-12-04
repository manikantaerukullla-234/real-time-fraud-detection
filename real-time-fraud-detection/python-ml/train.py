import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Separate input and output
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the ML model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")

print("✅ Fraud Model Trained Successfully!")
