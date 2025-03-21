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
print("📌 Original columns in dataset:", df.columns.tolist())

# Map the column names to be consistent with the Streamlit app
column_mapping = {
    "carrier_lg": "Airline",     # Airline column
    "nsmiles": "Distance",       # Distance in miles
    "fare": "Fare",              # Flight fare
    "city1": "Departure City",   # Departure location
    "city2": "Arrival City"      # Arrival location
}

# Check if expected columns exist
missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
if missing_columns:
    print(f"🚨 Missing columns: {missing_columns}. Please check the dataset format.")
    exit()

# Rename columns
df = df.rename(columns=column_mapping)

# Select only relevant columns (same as in Streamlit app)
selected_columns = ["Airline", "Distance", "Fare", "Departure City", "Arrival City"]
df = df[selected_columns]

# Convert numeric columns correctly
df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

# Handle missing values
print("Missing values before cleaning:\n", df.isnull().sum())
before_count = len(df)
df.dropna(inplace=True)  # Remove rows with missing values
after_count = len(df)
print("Missing values after cleaning:\n", df.isnull().sum())
print(f"Removed {before_count - after_count} rows with missing values")

# Store unique values for city selections (matching Streamlit app)
departure_cities = df["Departure City"].unique().tolist()
arrival_cities = df["Arrival City"].unique().tolist()
airlines = df["Airline"].unique().tolist()

print(f"Number of airlines: {len(airlines)}")
print(f"Number of departure cities: {len(departure_cities)}")
print(f"Number of arrival cities: {len(arrival_cities)}")

# Now apply encoding to categorical columns
label_encoders = {}
categorical_columns = ["Airline", "Departure City", "Arrival City"]
for column in categorical_columns:
    le = LabelEncoder()
    df[f"{column}_encoded"] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le  # Store encoders for later use

# Display the encoded data
print("\nEncoded data sample:")
print(df.head())

# Get basic statistics
print("\nBasic statistics:")
print(df.describe())

# Define X (features) and y (target) using the same format as Streamlit app
X = df[["Airline_encoded", "Distance", "Departure City_encoded", "Arrival City_encoded"]]
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

# Test prediction with a sample input using actual data
# Find a popular route to use as an example
popular_routes = df.groupby(["Departure City", "Arrival City"]).size().reset_index(name='count')
popular_routes = popular_routes.sort_values('count', ascending=False).head(5)
print("\nPopular routes:")
print(popular_routes)

if not popular_routes.empty:
    # Use the most popular route for prediction example
    sample_departure = popular_routes.iloc[0]["Departure City"]
    sample_arrival = popular_routes.iloc[0]["Arrival City"]
    
    # Get a common airline for this route
    common_airline = df[
        (df["Departure City"] == sample_departure) & 
        (df["Arrival City"] == sample_arrival)
    ]["Airline"].value_counts().idxmax()
    
    # Get average distance for this route
    avg_distance = df[
        (df["Departure City"] == sample_departure) & 
        (df["Arrival City"] == sample_arrival)
    ]["Distance"].mean()
    
    # Encode values
    airline_encoded = label_encoders["Airline"].transform([common_airline])[0]
    departure_encoded = label_encoders["Departure City"].transform([sample_departure])[0]
    arrival_encoded = label_encoders["Arrival City"].transform([sample_arrival])[0]
    
    # Make prediction
    sample_input = [[airline_encoded, avg_distance, departure_encoded, arrival_encoded]]
    predicted_price = model.predict(sample_input)
    
    print(f"\nSample route prediction:")
    print(f"Airline: {common_airline}")
    print(f"Route: {sample_departure} to {sample_arrival}")
    print(f"Distance: {avg_distance:.2f} miles")
    print(f"Predicted Price: ${predicted_price[0]:.2f}")
    
    # Get actual average price for comparison
    actual_avg = df[
        (df["Airline"] == common_airline) & 
        (df["Departure City"] == sample_departure) & 
        (df["Arrival City"] == sample_arrival)
    ]["Fare"].mean()
    
    print(f"Actual Average Price: ${actual_avg:.2f}")
    print(f"Prediction Difference: ${abs(predicted_price[0] - actual_avg):.2f} ({abs(predicted_price[0] - actual_avg)/actual_avg*100:.2f}%)")

# Create a visualization of feature relationships
plt.figure(figsize=(15, 10))

# Plot relationship between distance and fare
plt.subplot(2, 2, 1)
plt.scatter(df['Distance'], df['Fare'], alpha=0.3)
plt.title('Distance vs Fare')
plt.xlabel('Distance (miles)')
plt.ylabel('Fare ($)')

# Plot average fare by distance bins
plt.subplot(2, 2, 2)
df['Distance_bin'] = pd.cut(df['Distance'], bins=10)
distance_vs_fare = df.groupby('Distance_bin')['Fare'].mean().reset_index()
plt.bar(distance_vs_fare.index, distance_vs_fare['Fare'])
plt.xticks(rotation=45)
plt.title('Average Fare by Distance Range')
plt.xlabel('Distance Range')
plt.ylabel('Average Fare ($)')

# Plot average fare by top 10 airlines
plt.subplot(2, 2, 3)
top_airlines = df.groupby('Airline')['Fare'].mean().nlargest(10).sort_values(ascending=False)
top_airlines.plot(kind='bar')
plt.title('Average Fare by Top 10 Airlines')
plt.xlabel('Airline')
plt.ylabel('Average Fare ($)')
plt.xticks(rotation=45)

# Plot top 10 most expensive routes
plt.subplot(2, 2, 4)
route_fares = df.groupby(['Departure City', 'Arrival City'])['Fare'].mean().nlargest(10).reset_index()
route_labels = [f"{d[:3]}-{a[:3]}" for d, a in zip(route_fares['Departure City'], route_fares['Arrival City'])]
plt.bar(range(len(route_labels)), route_fares['Fare'])
plt.xticks(range(len(route_labels)), route_labels, rotation=45)
plt.title('Top 10 Most Expensive Routes')
plt.xlabel('Route')
plt.ylabel('Average Fare ($)')

plt.tight_layout()
plt.savefig('flight_price_analysis.png')
plt.show()

# Save the model and encoders for the Streamlit app to use
import pickle
with open('flight_price_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'encoders': label_encoders,
        'departure_cities': departure_cities,
        'arrival_cities': arrival_cities,
        'airlines': airlines
    }, f)
print("\n✅ Model and encoders saved for Streamlit app to use")