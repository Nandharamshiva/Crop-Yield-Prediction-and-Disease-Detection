import gdown
import os

# Google Drive file ID (from your shared folder link)
url = "https://drive.google.com/file/d/1dEtbZie8I45hR9wdmxXde_Gdy2lzHTG0/view?usp=sharing"
output = "yield_model.pkl"

# Download model if not already present
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)





import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -------------------------------
# Load Trained Models and Encoders
# -------------------------------
yield_model = pickle.load(open("yield_model.pkl", "rb"))
le_crop = pickle.load(open("le_crop.pkl", "rb"))
le_irrigation = pickle.load(open("le_irrigation.pkl", "rb"))

disease_model = tf.keras.models.load_model("plant_disease_model.h5")

# -------------------------------
# Utility Functions
# -------------------------------

def get_location():
    """Get user location using IP-based geolocation."""
    try:
        res = requests.get("https://ipinfo.io/").json()
        loc = res["loc"].split(",")
        latitude, longitude = float(loc[0]), float(loc[1])
        return latitude, longitude, res.get("city", ""), res.get("region", "")
    except:
        return None, None, None, None


def get_weather(lat, lon):
    """Fetch current weather data from OpenWeather API."""
    api_key = "9e1a234e8e8edb2ac4d359f82817f4b6"  # Replace with your own API key
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url).json()

        temp = response['main']['temp']
        humidity = response['main']['humidity']
        rainfall = response.get('rain', {}).get('1h', 0)

        weather = {
            "Temperature (°C)": temp,
            "Humidity (%)": humidity,
            "Rainfall (mm)": rainfall
        }
        return weather
    except:
        return {"Temperature (°C)": np.nan, "Humidity (%)": np.nan, "Rainfall (mm)": np.nan}


def predict_yield(crop, irrigation, area, prev_yield):
    """Predict yield using trained ML model (correct feature names)."""
    # Get location and weather
    lat, lon, city, state = get_location()
    weather = get_weather(lat, lon)

    # Average or placeholder soil and nutrient values (you can replace later with real inputs)
    soil_ph = 6.5
    soil_moisture = 25.0
    nitrogen = 1.2
    phosphorus = 0.9
    potassium = 1.1

    # Encode categorical variables
    crop_encoded = le_crop.transform([crop])[0]
    irrigation_encoded = le_irrigation.transform([irrigation])[0]

    # Prepare data with the same feature names as training
    data = pd.DataFrame([{
        'Area (acres)': area,
        'Crop_enc': crop_encoded,
        'Previous Year Yield (Kg per ha)': prev_yield,
        'Irrigation_enc': irrigation_encoded,
        'Average Temperature (°C)': weather['Temperature (°C)'],
        'Total Rainfall (mm)': weather['Rainfall (mm)'],
        'Humidity (%)': weather['Humidity (%)'],
        'Soil pH': soil_ph,
        'Soil Moisture (%)': soil_moisture,
        'Nitrogen (N) (%)': nitrogen,
        'Phosphorus (P) (%)': phosphorus,
        'Potassium (K) (%)': potassium
    }])

    # Predict using the model
    prediction = yield_model.predict(data)[0]

    return prediction, weather, city, state, lat, lon



def predict_disease(img_path):
    """Predict plant disease using CNN model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = disease_model.predict(img_array)
    class_idx = np.argmax(preds)

    disease_classes = [
    # 🍅 Tomato
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Healthy',

    # 🥔 Potato
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___Healthy',

    # 🍌 Banana
    'Banana___Black_sigatoka',
    'Banana___Healthy',

    # 🍎 Apple
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___Healthy',

    # 🍇 Grape
    'Grape___Black_rot',
    'Grape___Esca_Black_Measles',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot',
    'Grape___Healthy',

    # 🍊 Citrus
    'Citrus___Greening',
    'Citrus___Healthy',

    # 🌾 Corn (Maize)
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___Healthy',

    # 🍓 Strawberry
    'Strawberry___Leaf_scorch',
    'Strawberry___Healthy',

    # 🌶️ Pepper (Bell)
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___Healthy',

    # 🍠 Cassava
    'Cassava___Bacterial_Blight',
    'Cassava___Brown_Streak_Disease',
    'Cassava___Mosaic_Disease',
    'Cassava___Healthy',

    # 🫘 Other
    'Soybean___Healthy',
    'Squash___Powdery_mildew',
    'Blueberry___Healthy',
    'Raspberry___Healthy'
    ]

    return disease_classes[class_idx]


def get_precautions(disease_name):
    """Return precaution tips based on disease."""
    precautions = {
        # 🍅 Tomato
    'Tomato___Bacterial_spot': 'Use certified seeds, apply copper-based fungicide, and avoid overhead watering.',
    'Tomato___Early_blight': 'Use fungicides containing chlorothalonil or mancozeb; remove infected leaves.',
    'Tomato___Late_blight': 'Remove infected leaves, apply metalaxyl fungicide, and improve field drainage.',
    'Tomato___Leaf_Mold': 'Increase air circulation, reduce humidity, and apply copper-based sprays.',
    'Tomato___Septoria_leaf_spot': 'Remove debris, rotate crops, and apply fungicides like mancozeb.',
    'Tomato___Spider_mites_Two_spotted_spider_mite': 'Spray neem oil, avoid dry dusty conditions, and maintain field hygiene.',
    'Tomato___Target_Spot': 'Use fungicide sprays and avoid overhead irrigation.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies, remove infected plants, and use resistant varieties.',
    'Tomato___Tomato_mosaic_virus': 'Disinfect tools, wash hands, and use virus-free seeds.',
    'Tomato___Healthy': 'No disease detected. Maintain good irrigation and nutrient management.',

    # 🥔 Potato
    'Potato___Early_blight': 'Use certified seed, rotate crops, and apply fungicides containing chlorothalonil.',
    'Potato___Late_blight': 'Destroy infected plants, apply copper oxychloride spray, and avoid waterlogging.',
    'Potato___Healthy': 'No disease detected. Continue standard fertilizer and irrigation schedule.',

    # 🍌 Banana
    'Banana___Black_sigatoka': 'Prune affected leaves, improve air circulation, and spray propiconazole.',
    'Banana___Healthy': 'No disease detected. Keep the field clean and maintain soil fertility.',

    # 🍎 Apple
    'Apple___Apple_scab': 'Apply sulfur or copper fungicide before rainfall and prune infected branches.',
    'Apple___Black_rot': 'Remove mummified fruits, disinfect pruning tools, and use fungicide sprays.',
    'Apple___Cedar_apple_rust': 'Remove nearby juniper trees, spray myclobutanil fungicide, and prune infected areas.',
    'Apple___Healthy': 'No disease detected. Regular pruning and balanced fertilization recommended.',

    # 🍇 Grape
    'Grape___Black_rot': 'Remove infected leaves and berries, apply copper fungicides, and ensure air movement.',
    'Grape___Esca_Black_Measles': 'Avoid pruning during wet weather and remove infected vines.',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot': 'Use mancozeb spray and ensure adequate sunlight penetration.',
    'Grape___Healthy': 'No disease detected. Maintain pruning schedule and pest control.',

    # 🍊 Citrus
    'Citrus___Greening': 'Control psyllid insects with insecticides, remove infected trees, and use disease-free plants.',
    'Citrus___Healthy': 'No disease detected. Maintain adequate irrigation and fertilizer balance.',

    # 🌾 Corn (Maize)
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot': 'Use resistant hybrids, rotate crops, and apply fungicides early.',
    'Corn___Common_rust': 'Plant resistant varieties and apply strobilurin fungicides when necessary.',
    'Corn___Northern_Leaf_Blight': 'Use resistant seeds, apply mancozeb, and avoid overhead watering.',
    'Corn___Healthy': 'No disease detected. Keep soil well-drained and avoid overcrowding.',

    # 🍓 Strawberry
    'Strawberry___Leaf_scorch': 'Remove infected leaves, improve spacing, and apply captan fungicide.',
    'Strawberry___Healthy': 'No disease detected. Maintain proper spacing and irrigation.',

    # 🌶️ Pepper (Bell)
    'Pepper_bell___Bacterial_spot': 'Apply copper fungicide weekly and avoid working in wet fields.',
    'Pepper_bell___Healthy': 'No disease detected. Keep soil nutrients balanced.',

    # 🍠 Cassava
    'Cassava___Bacterial_Blight': 'Use resistant varieties and disinfect tools before pruning.',
    'Cassava___Brown_Streak_Disease': 'Use clean planting material and remove infected plants.',
    'Cassava___Mosaic_Disease': 'Control whiteflies and use disease-free cuttings.',
    'Cassava___Healthy': 'No disease detected. Continue normal irrigation.',

    # 🍅 Other
    'Soybean___Healthy': 'No disease detected. Maintain crop rotation and weed control.',
    'Squash___Powdery_mildew': 'Apply sulfur-based fungicide and ensure proper ventilation.',
    'Blueberry___Healthy': 'No disease detected. Maintain soil pH between 4.5–5.5 and avoid waterlogging.',
    'Raspberry___Healthy': 'No disease detected. Use drip irrigation and avoid overhead watering.'
    }
    return precautions.get(disease_name, "No specific precautions found.")


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Yield & Disease Detection", layout="wide")

st.title("🌾 Smart Farming Assistant")
st.markdown("Predict your **Crop Yield** and **Detect Plant Diseases** using AI and live weather 🌦️")

tab1, tab2 = st.tabs(["🌱 Crop Yield Prediction", "🍃 Plant Disease Detection"])

# ---------------------------------
# TAB 1: Crop Yield Prediction
# ---------------------------------
with tab1:
    st.header("🌾 Crop Yield Prediction")

    crop = st.selectbox("Select Crop", le_crop.classes_)
    irrigation = st.selectbox("Select Irrigation Type", le_irrigation.classes_)
    area = st.number_input("Enter Land Area (in acres)", min_value=0.1)
    prev_yield = st.number_input("Previous Year Yield (Kg per ha)", min_value=0.0)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Yield"):
            with st.spinner("Fetching location and weather data..."):
                prediction, weather, city, state, lat, lon = predict_yield(crop, irrigation, area, prev_yield)

            # Show map and weather info
            st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
            st.success(f"📍 Location: {city}, {state}")
            st.info(f"🌡️ Temp: {weather['Temperature (°C)']}°C | 💧 Humidity: {weather['Humidity (%)']}% | 🌧️ Rainfall: {weather['Rainfall (mm)']} mm")
            st.subheader(f"✅ Predicted Yield: {prediction:.2f} Kg per ha")

    with col2:
        if st.button("Check Live Weather"):
            lat, lon, city, state = get_location()
            weather = get_weather(lat, lon)
            st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
            st.success(f"📍 Location: {city}, {state}")
            st.info(f"🌡️ Temp: {weather['Temperature (°C)']}°C | 💧 Humidity: {weather['Humidity (%)']}% | 🌧️ Rainfall: {weather['Rainfall (mm)']} mm")

# ---------------------------------
# TAB 2: Plant Disease Detection
# ---------------------------------
with tab2:
    st.header("🍃 Plant Disease Detection")

    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Disease"):
            with st.spinner("Analyzing leaf image..."):
                disease_name = predict_disease(uploaded_file)
                precaution = get_precautions(disease_name)

            st.success(f"🌿 Disease Detected: **{disease_name}**")
            st.info(f"🩺 Precaution: {precaution}")
