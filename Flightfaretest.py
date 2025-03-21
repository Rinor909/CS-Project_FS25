import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Install dependencies (only run once)
import gdown
import pandas as pd

# Google Drive file ID
file_id = "11NgU1kWQIAzBhEbG3L6XsLRqm1T2dn4I"

# File name to save as
output = "US_Airline_Flight_Routes_and_Fares.csv"

# Download the file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load dataset into pandas
df = pd.read_csv(output)

# Display first few rows
print(df.head())

# Show missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])

# Drop rows with missing values
df.dropna(inplace=True)

# Verify missing values are gone
print("Missing values after cleaning:\n", df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Convert 'Duration' from '6h 15m' to total minutes
def convert_duration(duration):
    try:
        h, m = map(int, duration.replace("h", "").replace("m", "").split())
        return h * 60 + m  # Convert to total minutes
    except:
        return None  # Handle invalid values

df["Duration"] = df["Duration"].apply(convert_duration)
df.dropna(subset=["Duration"], inplace=True)  # Remove invalid rows

# Encode categorical columns
label_encoders = {}
for column in ["Airline", "Departure Time", "Class"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoders for later use

print(df.head())  # Check the cleaned data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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