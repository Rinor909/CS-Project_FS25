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

# Download dataset from Google Drive
if not os.path.exists(output):
    print("⚠️ Downloading dataset from Google Drive...")
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    with open(output, "wb") as file:
        file.write(response.content)
    print("✅ File downloaded successfully!")
else:
    print("✅ File already exists, skipping download.")

# Load dataset
df = pd.read_csv(output)  # Load the dataset

# Print available columns for debugging
print("📌 Columns in dataset:", df.columns.tolist())

# Ensure all expected columns exist
expected_columns = ["Airline", "Distance", "Departure Time", "Duration", "Class", "Fare"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"🚨 Missing columns: {missing_columns}. Please check the dataset format.")
    exit()

# Convert 'Duration' from '6h 15m' to total minutes
def convert_duration(duration):
    if pd.isna(duration):
        return None
    try:
        duration = str(duration)
        if 'h' in duration and 'm' in duration:
            parts = duration.split()
            hours = int(parts[0].replace("h", ""))
            minutes = int(parts[1].replace("m", ""))
        elif 'h' in duration:
            hours = int(duration.replace("h", ""))
            minutes = 0
        elif 'm' in duration:
            hours = 0
            minutes = int(duration.replace("m", ""))
        else:
            return None
        return hours * 60 + minutes
    except Exception as e:
        print(f"Error converting duration '{duration}': {e}")
        return None

df["Duration"] = df["Duration"].apply(convert_duration)

# Convert numeric columns correctly
df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

# Handle missing values
print("Missing values before cleaning:\n", df.isnull().sum())
df.dropna(inplace=True)  # Remove rows with missing values
print("Missing values after cleaning:\n", df.isnull().sum())

# Now apply encoding to categorical columns
label_encoders = {}
categorical_columns = ["Airline", "Departure Time", "Class"]
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le  # Store encoders for later use

# Display the encoded data
print("\nEncoded data sample:")
print(df.head())

# Get basic statistics
print("\nBasic statistics:")
print(df.describe())

# Define X (features) and y (target)
X = df[["Airline", "Distance", "Departure Time", "Duration", "Class"]]
y = df["Fare"]

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nModel R² score on training data: {train_score:.4f}")
print(f"Model R² score on test data: {test_score:.4f}")

# Feature importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Test prediction with a sample input
# Note: You would need to use the same encoding as during training
sample_airline = 2  # Example airline code
sample_distance = 1500
sample_departure = 1  # Example departure time code
sample_duration = 360  # 6 hours in minutes
sample_class = 0  # Example class code

sample_input = [[sample_airline, sample_distance, sample_departure, sample_duration, sample_class]]
predicted_price = model.predict(sample_input)
print(f"\nPredicted Flight Price: ${predicted_price[0]:.2f}")

# Create a visualization of feature relationships
plt.figure(figsize=(12, 8))

# Plot relationship between distance and fare
plt.subplot(2, 2, 1)
plt.scatter(df['Distance'], df['Fare'], alpha=0.5)
plt.title('Distance vs Fare')
plt.xlabel('Distance (miles)')
plt.ylabel('Fare ($)')

# Plot relationship between duration and fare
plt.subplot(2, 2, 2)
plt.scatter(df['Duration'], df['Fare'], alpha=0.5)
plt.title('Duration vs Fare')
plt.xlabel('Duration (minutes)')
plt.ylabel('Fare ($)')

# Plot average fare by airline
plt.subplot(2, 2, 3)
avg_fare_by_airline = df.groupby('Airline')['Fare'].mean().sort_values(ascending=False)
avg_fare_by_airline.plot(kind='bar')
plt.title('Average Fare by Airline')
plt.xlabel('Airline Code')
plt.ylabel('Average Fare ($)')

# Plot average fare by class
plt.subplot(2, 2, 4)
avg_fare_by_class = df.groupby('Class')['Fare'].mean().sort_values(ascending=False)
avg_fare_by_class.plot(kind='bar')
plt.title('Average Fare by Class')
plt.xlabel('Class Code')
plt.ylabel('Average Fare ($)')

plt.tight_layout()
plt.savefig('flight_price_analysis.png')
plt.show()