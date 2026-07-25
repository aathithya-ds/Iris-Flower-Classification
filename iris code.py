import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load dataset
df = pd.read_csv("iris (2).csv")

print("Missing values:")
print(df.isnull().sum())

# Handle missing values
df = df.dropna(subset=["species"])

df["sepal_length"] = df["sepal_length"].fillna(0)
df["sepal_width"] = df["sepal_width"].fillna(df["sepal_width"].mean())
df["petal_length"] = df["petal_length"].fillna(df["petal_length"].median())
df["petal_width"] = df["petal_width"].fillna(df["petal_width"].mean())

# Encode target column
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)
loss = log_loss(y_test, probabilities)

print("\nResults")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Log Loss: {loss:.4f}")

# Save model
with open("iris_Logistic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save label encoder
with open("iris_label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(le, encoder_file)

print("\nModel and Label Encoder saved successfully!")

# User prediction
a1 = float(input("Enter sepal_length: "))
a2 = float(input("Enter sepal_width: "))
a3 = float(input("Enter petal_length: "))
a4 = float(input("Enter petal_width: "))

sample = [[a1, a2, a3, a4]]

prediction = model.predict(sample)

# Convert encoded value back to flower name
flower_name = le.inverse_transform(prediction)

print("\nPredicted Species:", flower_name[0])