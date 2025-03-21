import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Google Drive file ID (Extracted from shared link)
file_id = "11NgU1kWQIAzBhEbG3L6XsLRqm1T2dn4I"
output = "US_Airline_Flight_Routes_and_Fares.csv"

# Download dataset from Google Drive (Using requests instead of gdown)
if not os.path.exists(output):
    print("⚠️ Downloading dataset from Google Drive...")
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    with open(output, "wb") as file:
        file.write(response.content)
    print("✅ File downloaded successfully!")

# Load dataset
df = pd.read_csv(output, dtype=str)  # Load everything as string first

# Print available columns for debugging
print("📌 Columns in dataset:", df.columns.tolist())

# Ensure all expected columns exist
expected_columns = ["Airline", "Distance", "Departure Time", "Duration", "Class", "Fare"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"🚨 Missing columns: {missing_columns}. Please check the dataset format.")
    exit()

# Convert categorical columns
categorical_columns = ["Airline", "Departure Time", "Class"]
for col in categorical_columns:
    df[col] = df[col].astype("category").cat.codes

# Convert 'Duration' from '6h 15m' to total minutes
def convert_duration(duration):
    try:
        parts = duration.split(" ")
        hours = int(parts[0].replace("h", ""))
        minutes = int(parts[1].replace("m", "")) if len(parts) > 1 else 0
        return hours * 60 + minutes
    except:
        return None  # Handle invalid values

df["Duration"] = df["Duration"].apply(convert_duration)

# Convert numeric columns correctly
numeric_cols = ["Distance", "Duration", "Fare"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert, replace errors with NaN

df.dropna(inplace=True)  # Remove invalid rows

# Display first few rows
print(df.head())

# Show missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])

# Encode categorical columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoders for later use

print(df.head())  # Check the cleaned data

# Define X (features) and y (target)
X = df[["Airline", "Distance", "Departure Time", "Duration", "Class"]]
y = df["Fare"]

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
sample_input = [[2, 1500, 1, 360, 0]]  # Example: Airline 2, 1500 miles, Morning, 6-hour duration, Economy class
predicted_price = model.predict(sample_input)
print(f"Predicted Flight Price: ${predicted_price[0]:.2f}")