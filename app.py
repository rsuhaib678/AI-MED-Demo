import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# Load Models
#brain_tumor_model = load_model('models/brain_tumor_model.h5', compile=False)
lung_cancer_model = load_model('models/lung_cancer_model.h5', compile=False)
eye_disease_model = load_model('models/eye_disease_model.h5', compile=False)
breast_cancer_model = load_model('models/breast_cancer_model.h5', compile=False)
heart_disease_model = joblib.load('models/heart_disease_model.pkl')

# Load Scalers
breast_cancer_scaler = joblib.load('models/breast_cancer_scaler.pkl')

# Class Names
lung_cancer_classes = ['Benign', 'Malignant', 'Normal']
eye_disease_classes = ['Glaucoma', 'Cataract', 'Diabetic Retinopathy', 'Macular Degeneration']
brain_tumor_classes = ['Benign Tumor', 'Malignant Tumor']
breast_cancer_classes = ['Malignant', 'Benign']
heart_disease_classes = ['No Disease', 'Disease']

# Preprocessing Functions
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize to match model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def preprocess_tabular(data, scaler=None):
    data = np.array(data).reshape(1, -1)  # Reshape for single sample
    if scaler:
        data = scaler.transform(data)
    return data

# Streamlit App Layout
st.set_page_config(page_title="AI-MED Models UK", page_icon="🩺")
st.title("Welcome to AI-MED Models UK")
st.image("images/logo.png", width=150)
st.markdown(
    "<h3 style='text-align: center; color: #4CAF50;'>"
    "<a href='http://www.aimedmodels.com' target='_blank'>Visit AI-MED Models</a>"
    "</h3>",
    unsafe_allow_html=True
)

# File Upload for Image-based Models
image_file = st.file_uploader("Upload an Image (for Lung Cancer, Eye Disease, or Brain Tumor Detection)", type=["jpg", "png", "jpeg"])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    model_choice = st.selectbox("Select the model to use for prediction:", ["Lung Cancer", "Eye Disease", "Brain Tumor"])

    if model_choice == "Lung Cancer":
        img_array = preprocess_image(image)
        prediction = lung_cancer_model.predict(img_array)
        predicted_class = lung_cancer_classes[np.argmax(prediction)]
        st.write(f"Lung Cancer Prediction: **{predicted_class}**")

    elif model_choice == "Eye Disease":
        img_array = preprocess_image(image)
        prediction = eye_disease_model.predict(img_array)
        predicted_class = eye_disease_classes[np.argmax(prediction)]
        st.write(f"Eye Disease Prediction: **{predicted_class}**")

    """elif model_choice == "Brain Tumor":
        img_array = preprocess_image(image)
        prediction = brain_tumor_model.predict(img_array)
        predicted_class = brain_tumor_classes[np.argmax(prediction)]
        st.write(f"Brain Tumor Prediction: **{predicted_class}**")"""

# Tabular Input for Heart and Breast Cancer Models
st.subheader("Heart Disease Detection")
heart_inputs = [
    st.number_input("Age", min_value=0, max_value=120, value=50),
    st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
    st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3]),
    st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120),
    st.number_input("Cholesterol", min_value=100, max_value=400, value=200),
    st.selectbox("Fasting Blood Sugar > 120 (0 = No, 1 = Yes)", [0, 1]),
    st.selectbox("Resting ECG Results (0-2)", [0, 1, 2]),
    st.number_input("Max Heart Rate Achieved", min_value=60, max_value=200, value=150),
    st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1]),
    st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=5.0, value=0.0, step=0.1),
    st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2]),
    st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3]),
    st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])
]
if st.button("Predict Heart Disease"):
    heart_data = preprocess_tabular(heart_inputs)
    prediction = heart_disease_model.predict(heart_data)
    predicted_class = heart_disease_classes[int(prediction[0])]
    st.write(f"Heart Disease Prediction: **{predicted_class}**")

st.subheader("Breast Cancer Detection")
breast_inputs = [st.number_input(f"{feature}", value=0.0) for feature in [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean",
    "Fractal Dimension Mean"
]]
if st.button("Predict Breast Cancer"):
    breast_data = preprocess_tabular(breast_inputs, scaler=breast_cancer_scaler)
    prediction = breast_cancer_model.predict(breast_data)
    predicted_class = breast_cancer_classes[np.argmax(prediction)]
    st.write(f"Breast Cancer Prediction: **{predicted_class}**")

# Add Styling
st.markdown("""
    <style>
        .stApp {background-color: #f4f4f9;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
        .stButton>button:hover {background-color: #45a049;}
    </style>
""", unsafe_allow_html=True)
