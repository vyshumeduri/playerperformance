# playerperformance
SEVEN7CODE TECHNOLOGIES-DATA SCIENCE CODE

Creating a cricket player performance prediction system involves similar steps as the Titanic classification. Here's a simplified example using the dataset you provided. We'll use a RandomForestRegressor this time, as we're predicting a continuous variable (performance score).
python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Download the dataset from the provided URL
url = "https://www.kaggle.com/datasets/saivamshi/cricket-world-cup-2019-players-data"
print("Please download the dataset from:", url)
print("After downloading, save it as 'cricket-players-data.csv' in the same directory as this script.")
input("Press Enter to continue once you've downloaded the dataset...")

# Load the dataset
data = pd.read_csv("cricket-players-data.csv")

# Data preprocessing
# Handle missing values, feature selection, and encoding if needed

# Selecting features and target variable
X = data[['Matches', 'Innings', 'Runs', 'Avg', 'SR', 'Wkts', 'Econ', 'Ct', 'St']]
y = data['Performance']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error (a common regression metric)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
