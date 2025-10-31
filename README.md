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

Farmers only need to provide:
-  Crop name  
-  Land area (in acres)  
-  Irrigation type  
-  Previous year yield  

Everything else — including weather, soil, and environmental factors — is fetched automatically using APIs.  

Additionally, users can **upload or capture plant leaf images** to detect diseases and receive **precautionary suggestions** retrieved directly from Google.

## Features
- Crop yield prediction based on area, crop type, and irrigation method  
- Automatic weather and location detection  
- Disease detection via image upload or camera capture  
- Real-time precautions from Google  
- User-friendly Streamlit web app interface  

##  Dataset Details
1. Crop Yield Prediction Dataset 
   - Includes data on crop, soil type, nutrients, rainfall, temperature, humidity, and past yield.  
   - Used to train the yield prediction model.

2. PlantVillage Dataset
   - Contains labeled images of healthy and diseased plant leaves.  
   - Used to train the CNN-based disease detection model.

Download Datasets & Trained Models:
Google Drive Folder : (https://drive.google.com/drive/folders/1DGX10A0xa06CAtnFFmjcxd7Q6SPjYuzO?usp=sharing)

## Models
- "yield_model.pkl" → XGBoost model for yield prediction  
- "plant_disease_model.h5" → CNN model for disease detection  

##  Run Locally
To run this project on your system:
-- bash
pip install -r requirements.txt
streamlit run app.py

or git clone https://github.com/Nandharamshiva/Crop-Yield-Prediction-and-Disease-Detection.git
   cd Crop-Yield-Prediction-and-Disease-Detection
