import joblib
import streamlit as st
import pandas as pd

## Load the trained model and the list of features it expects
model = joblib.load("laptop_price_model.pkl")
feature_names = joblib.load("model_features.pkl")

## App title and short description
st.title("Laptop Price Estimator (SGD)")
st.write(
    "Estimate a laptop's price in Singapore dollars from its specifications. "
    "This tool is aimed at online sellers and small retailers to help them "
    "price laptops more consistently."
)

## ---- User inputs ----
st.header("Enter the laptop specifications")

## Categorical inputs (dropdowns) - the options must match what the model was trained on
company = st.selectbox(
    "Brand",
    ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Razer', 'Apple', 'Other']
)
type_name = st.selectbox(
    "Laptop type",
    ['Notebook', 'Gaming', 'Ultrabook', 'Workstation', '2 in 1 Convertible', 'Netbook']
)
cpu_tier = st.selectbox(
    "CPU tier",
    ['Core i3', 'Core i5', 'Core i7', 'Other']
)
gpu_brand = st.selectbox(
    "GPU brand",
    ['Intel', 'Nvidia', 'AMD']
)
os_choice = st.selectbox(
    "Operating system",
    ['Windows', 'Mac', 'No OS']
)

## Numeric inputs (sliders / number inputs)
ram = st.selectbox("RAM (GB)", [2, 4, 8, 12, 16, 24, 32, 64])
inches = st.slider("Screen size (inches)", 10.0, 18.0, 15.6, step=0.1)
weight = st.slider("Weight (kg)", 0.7, 4.5, 1.8, step=0.1)
cpu_ghz = st.slider("CPU speed (GHz)", 0.9, 3.6, 2.5, step=0.1)
ppi = st.slider("Screen sharpness (PPI)", 90, 350, 141)
ssd = st.selectbox("SSD storage (GB)", [0, 128, 256, 512, 1024])
hdd = st.selectbox("HDD storage (GB)", [0, 500, 1024, 2048])
touchscreen = st.checkbox("Touchscreen")
ips = st.checkbox("IPS display")


## ---- Prediction ----
if st.button("Estimate price"):

    ## Start with a row of all zeros for every feature the model expects
    row = {f: 0 for f in feature_names}

    ## Fill in the numeric features
    row['Ram_GB']     = ram
    row['Weight_kg']  = weight
    row['PPI']        = ppi
    row['Cpu_GHz']    = cpu_ghz
    row['SSD_GB']     = ssd
    row['HDD_GB']     = hdd
    row['Inches']     = inches
    row['Touchscreen'] = 1 if touchscreen else 0
    row['IPS']         = 1 if ips else 0

    ## Set the correct one-hot column to 1 (only if that column exists in the model)
    ## The dropped/first category is represented by all-zeros, so we use .get-style checks
    for col in [f"Company_{company}", f"TypeName_{type_name}",
                f"Cpu_tier_{cpu_tier}", f"Gpu_brand_{gpu_brand}",
                f"OS_{os_choice}"]:
        if col in row:
            row[col] = 1

    ## Build a one-row DataFrame in the exact column order the model expects
    X_input = pd.DataFrame([row])[feature_names]

    ## Predict
    predicted_price = model.predict(X_input)[0]

    st.success(f"Estimated price: S${predicted_price:,.2f}")
    st.caption(
        "This is an estimate based on specifications only. Real prices may vary "
        "due to brand premium, discounts, and demand."
    )