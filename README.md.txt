#  Crop Yield Prediction and Disease Detection using Machine Learning

##  Overview
This project combines Machine Learning and Deep Learning to help farmers predict crop yield and detect plant diseases. It automatically collects weather and location data for accurate yield forecasting and provides real-time precautionary advice using Google search results.

## Technologies Used
- Python  
- Pandas, NumPy, Scikit-learn  
- TensorFlow / Keras  
- Streamlit  
- OpenCV  
- Geopy, Weather API  
- Google Custom Search API  

## Features
- Crop yield prediction based on area, crop type, and irrigation method  
- Automatic weather and location detection  
- Disease detection via image upload or camera capture  
- Real-time precautions from Google  
- User-friendly Streamlit web app interface  

## Models
- "yield_model.pkl" → XGBoost model for yield prediction  
- "plant_disease_model.h5" → CNN model for disease detection  

##  Run Locally
To run this project on your system:
```bash
pip install -r requirements.txt
streamlit run app.py
