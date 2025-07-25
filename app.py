import streamlit as st
import requests

st.title("IRIS Flower Classifier - FastAPI Client")

st.write("Enter the features of the IRIS")

# Input form

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)  


if st.button("Predict"):
    # Prepare the data
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    
    # Make the request to the FastAPI server
    response = requests.post("http://localhost:8000/predict", json=data)
    
    if response.status_code == 200:
        prediction = response.json()
            # Display result
        st.success(f"Predicted Species: {prediction['species_name']} (Class {prediction['prediction']})")

    else:
        st.error("Error in prediction")