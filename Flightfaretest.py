import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from Kaggle
df = pd.read_csv(
    '/kaggle/input/us-airline-flight-routes-and-fares-1993-2024/US Airline Flight Routes and Fares 1993-2024.csv',
    low_memory=False
)

from sklearn.preprocessing import LabelEncoder

# Convert "Duration" from '6h 15m' to minutes
def convert_duration(duration):
    h, m = map(int, duration.replace("h", "").replace("m", "").split())
    return h * 60 + m

df["Duration"] = df["Duration"].apply(convert_duration)

# Encode categorical columns
le = LabelEncoder()
df["Airline"] = le.fit_transform(df["Airline"])
df["Class"] = le.fit_transform(df["Class"])
df["Departure Time"] = le.fit_transform(df["Departure Time"])

# Drop unnecessary columns
df = df.drop(columns=["Date"])

print(df.head())  # Check cleaned data


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define X (independent variables) and y (target variable)
X = df[["Airline", "Distance", "Departure Time", "Duration", "Class"]]
y = df["Fare"]

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
sample_input = [[2, 1500, 1, 360, 0]]  # Example: Airline 2, 1500 miles, Morning flight, 6-hour duration, Economy class
predicted_price = model.predict(sample_input)
print(f"Predicted Flight Price: ${predicted_price[0]:.2f}")