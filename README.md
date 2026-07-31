# Laptop Price Estimator — Streamlit App

**Name:** Harel Tan Jun Long  
**Admin No:** 2504542H  

A web based app that utilizes a laptop's spec to predict its price in SGD (Singapore dollars). Using a tuned random forest model which received training in the project notebook (ipynb file). The web based app was developed with Streamlit designed for assisting small retailers and online sellers in determining consisten laptop prices

## Features
- In the sidebar there is a form for the user can fill up with the laptop's specs (brand, type, RAM, storage, CPU, GPU and etc.) to find the laptop price
- after the laptop finishes calculating it will give a predicted price with a lower and upper range (typical error (MAE) of S$286.97)
- Visualisations:
  - Price vs RAM and Price vs SSD (price responds to each spec)
  - Top factors influencing the predicted price
  - Suggested pricing range
- Input validation and user-facing error messages
- Prediction history table with CSV download

## Files
- `app.py`: the Streamlit application
- `laptop_price_model.pkl`: the trained model (loaded by app.py)
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: app theme

## Model
The model is a tuned Random Forest Regressor trained on the laptop price dataset. It predicts `Price_SGD`
from laptop specifications with a typical error (MAE) of S$286.97
