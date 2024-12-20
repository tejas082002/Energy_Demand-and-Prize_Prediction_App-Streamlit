import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Energy_Demand-and-Prize_Prediction_App-Streamlit-main\Data\EW.csv")

# Energy Demand Prediction
# Select the independent variables and the target variable
X_demand = df[['generation fossil brown coal/lignite',
               'generation fossil gas', 'generation fossil hard coal', 'generation fossil oil','total load actual']]
y_demand = df['total load forecast']

# Split the data into training and testing sets
X_demand_train, X_demand_test, y_demand_train, y_demand_test = train_test_split(X_demand, y_demand, test_size=0.2,
                                                                                random_state=42)

# Create and train the Linear Regression model for Energy Demand Prediction
demand_model = LinearRegression()
demand_model.fit(X_demand_train, y_demand_train)

# Energy Price Prediction
# Select the independent variables and the target variable
X_price = df[['price actual', 'generation fossil brown coal/lignite',
              'generation fossil gas', 'generation fossil hard coal', 'generation fossil oil']]
y_price = df['price day ahead']

# Split the data into training and testing sets
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2,
                                                                            random_state=42)

# Create and train the Linear Regression model for Energy Price Prediction
price_model = LinearRegression()
price_model.fit(X_price_train, y_price_train)

# Energy Weather Prediction
# Select the independent variables and the target variable
X_weather = df[['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg']]
y_weather = df['weather_main']

# Split the data into training and testing sets
X_weather_train, X_weather_test, y_weather_train, y_weather_test = train_test_split(X_weather, y_weather, test_size=0.2,
                                                                                    random_state=42)

# Create and train the Linear Regression model for Weather Prediction
weather_model = LogisticRegression()
weather_model.fit(X_weather_train, y_weather_train)

# Set the app title
st.title("Energy Demand & Prize Prediction")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Oil & Gas Industry Prediction System',

                           ['Energy Demand',
                            'Energy Prize',
                            'Weather'],
                           icons=['bolt','dollar','cloud'],
                           default_index=0)

if selected == "Energy Demand":
    st.header("Energy Demand Prediction")

    # Get user input for the independent variables using text boxes
    generation_brown_coal_demand = st.number_input("Generation from brown coal/lignite (Demand)", key="demand_brown_coal")
    generation_gas_demand = st.number_input("Generation from fossil gas (Demand)", key="demand_gas")
    generation_hard_coal_demand = st.number_input("Generation from hard coal (Demand)", key="demand_hard_coal")
    generation_oil_demand = st.number_input("Generation from fossil oil (Demand)", key="demand_oil")
    total_load_actual_demand = st.number_input("Total load actual (Demand)", key="demand_total_load")

    # Button to trigger prediction calculation
    if st.button("Predict Energy Demand"):
        # Predict the energy demand
        demand_prediction = demand_model.predict(
            [[generation_brown_coal_demand, generation_gas_demand,
              generation_hard_coal_demand, generation_oil_demand,total_load_actual_demand]])

        # Display the predicted energy demand inside a box
        st.success(f"Predicted total load forecast (Energy Demand): {int(demand_prediction[0])}")

elif selected == "Energy Prize":
    st.header("Energy Price Prediction")

    # Get user input for the independent variables using text boxes
    generation_brown_coal_price = st.number_input("Generation from brown coal/lignite (Price)", key="price_brown_coal")
    generation_gas_price = st.number_input("Generation from fossil gas (Price)", key="price_gas")
    generation_hard_coal_price = st.number_input("Generation from hard coal (Price)", key="price_hard_coal")
    generation_oil_price = st.number_input("Generation from fossil oil (Price)", key="price_oil")
    price_actual = st.number_input("Actual price (Price)", key="price_actual")

    # Button to trigger prediction calculation
    if st.button("Predict Energy Price"):
        # Predict the energy price
        price_prediction = price_model.predict([[price_actual, generation_brown_coal_price,
                                                 generation_gas_price, generation_hard_coal_price, generation_oil_price]])

        # Display the predicted energy price inside a box
        st.success(f"Predicted price day ahead (Energy Price): {price_prediction[0]}")

elif selected == "Weather":
    st.header("Weather Prediction")

    # Get user input for the independent variables using text boxes
    temp = st.number_input("Temperature", key="weather_temp")
    pressure = st.number_input("Pressure", key="weather_pressure")
    humidity = st.number_input("Humidity", key="weather_humidity")
    wind_speed = st.number_input("Wind Speed", key="weather_wind_speed")
    wind_deg = st.number_input("Wind Degree", key="weather_wind_deg")

    # Button to trigger prediction calculation
    if st.button("Predict Weather"):
        # Predict the weather category
        weather_prediction = weather_model.predict([[temp, pressure, humidity, wind_speed, wind_deg]])

        # Display the predicted weather category inside a box
        st.success(f"Predicted Weather Category: {weather_prediction[0]}")

