import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load and preprocess the weather data for a given city
def load_and_preprocess_data(city):
    # Load the CSV file for the specified city and handle any missing values
    df = pd.read_csv(f"{city}.csv", header=0, na_values=['-'], encoding='utf-8-sig')
    # Set the 'Year' column as the index
    df.set_index('Year', inplace=True)
    # Filter the DataFrame for the years 1980 to 2017
    df = df.loc[1980:2017]
    # Interpolate missing data
    df.interpolate(inplace=True)
    # Reset the index to convert 'Year' back to a column
    df.reset_index(inplace=True)
    return df

# Function to predict weather data based on the processed DataFrame and target variable
def predict_weather(df, targetdata):
    # Define features (independent variable) and target (dependent variable)
    features = 'Year'
    target = targetdata
    # Convert features and target to 2D arrays
    x = df[features].values.reshape(-1, 1)  # Feature (Year)
    y = df[target].values.reshape(-1, 1)    # Target (e.g., Avg Temp)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)

    # Create and train the linear regression model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predict values for the test set
    y_prediction = regressor.predict(x_test)
    print("\nPredicted Values for the Test Set:")
    print(y_prediction.flatten())

    # Calculate slope and intercept for the regression line
    slope, intercept = np.polyfit(df[features], df[target], 1)

    # Prompt user for a year to predict the target variable
    year = float(input('Enter the year to predict the value: '))
    # Calculate predicted value using the linear regression formula
    temp = intercept + (slope * year)
    print(f"Predicted {targetdata} for the year {int(year)}: {temp:.2f}\n")

    # Plot the results
    plt.scatter(x_test, y_test, color='black')  # Scatter plot for actual test values
    plt.plot(x_test, slope * x_test + intercept, '-', color='blue')  # Line plot for predicted values
    plt.title(f'Prediction for {targetdata} over Years')
    plt.xlabel('Year')
    plt.ylabel(targetdata)
    plt.grid()  # Add grid for better readability
    plt.show()  # Show the plot

# Function to manage prediction for a specified city
def run_city_prediction(city):
    # Load the processed data for the city
    df = load_and_preprocess_data(city)
    while True:
        # Display options for target data to predict
        print("\nChoose target data to predict:")
        print("1. Average Temperature (T)")
        print("2. Total Rainfall (PP)")
        print("3. Average Windspeed (V)")
        print("4. Number of Days with Rain (RA)")
        print("5. Number of Days with Fog (FG)")
        print("6. Exit")

        # Get user choice for target data
        targetchoice = int(input("Enter your choice: "))

        # Call the prediction function based on user choice
        if targetchoice == 1:
            predict_weather(df, 'T')  # Average Temperature
        elif targetchoice == 2:
            predict_weather(df, 'PP')  # Total Rainfall
        elif targetchoice == 3:
            predict_weather(df, 'V')   # Average Windspeed
        elif targetchoice == 4:
            predict_weather(df, 'RA')  # Number of Days with Rain
        elif targetchoice == 5:
            predict_weather(df, 'FG')  # Number of Days with Fog
        elif targetchoice == 6:
            break  # Exit the loop
        else:
            print("Invalid Choice, please try again.")

# Main function to control the flow of the program
def main():
    print("\n\n\t\t\t\t\t\tWelcome to Weather Prediction\n")
    
    while True:
        # Display options for cities
        print("Choose a city for weather prediction:")
        print("1. Bangalore")
        print("2. Mumbai")
        print("3. Delhi")
        print("4. Chennai")
        print("5. Kolkata")
        print("6. Exit")
        
        # Get user choice for city
        choice = int(input("Enter your choice: "))

        # Call the city prediction function based on user choice
        if choice == 1:
            run_city_prediction("bangalore")
        elif choice == 2:
            run_city_prediction("mumbai")
        elif choice == 3:
            run_city_prediction("delhi")
        elif choice == 4:
            run_city_prediction("chennai")
        elif choice == 5:
            run_city_prediction("kolkata")
        elif choice == 6:
            print("Exiting the program. Goodbye!")
            break  # Exit the program
        else:
            print("Invalid Choice, please try again.")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
