
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Food Waste Predictor")
st.write("Predict how many meals might go to waste at restaurants.")

# Sample data
data = {
    "Restaurant": ["Aroma Café", "Burger Hub", "Fresh Mart", "Pizza Point", "Green Deli"],
    "Meals_Prepared": [120, 150, 200, 180, 160],
    "Meals_Sold": [100, 130, 150, 160, 140]
}
df = pd.DataFrame(data)
df["Meals_Wasted"] = df["Meals_Prepared"] - df["Meals_Sold"]

st.write("Current food waste data:")
st.dataframe(df)

# Train model
X = df[["Meals_Prepared"]]
y = df["Meals_Wasted"]
model = LinearRegression()
model.fit(X, y)

# Input from user
meals_prepared = st.number_input("Enter meals prepared:", min_value=0, value=150)
predicted_waste = model.predict([[meals_prepared]])
st.write("Predicted meals wasted:", int(predicted_waste[0]))
