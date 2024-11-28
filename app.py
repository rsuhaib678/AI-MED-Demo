import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import pandas as pd

# Load Models
brain_tumor_model = load_model('models/brain_tumor_model.h5', compile=False)
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
    """Preprocess images for input into models."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def preprocess_tabular(data, scaler=None):
    """Preprocess tabular data for input into models."""
    data = np.array(data).reshape(1, -1)
    if scaler:
        data = scaler.transform(data)
    return data

# App Configuration
st.set_page_config(page_title="AI-MED Models UK", page_icon="🩺", layout="wide")

# CSS Styling for Improved Layout
st.markdown(
    """
    <style>
    /* App Background */
    .stApp {
        background-color: #f4f8fc;
        padding: 20px;
    }

    /* Header Section */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 50px;
    }

    .header-logo {
        width: 120px;
        height: auto;
    }

    .header-text {
        text-align: center;
        flex-grow: 1;
    }

    .header-title {
        font-size: 55px;
        color: #004aad;
        font-weight: bold;
    }

    .header-subtitle {
        font-size: 22px;
        font-style: italic;
        color: #008cba;
        margin-top: -10px;
    }

    .header-link {
        font-size: 18px;
        color: #004aad;
        text-decoration: none;
        font-weight: bold;
    }

    .header-link:hover {
        color: #0077b6;
        text-decoration: underline;
    }

    /* Custom Tabs Styling */
    .custom-tabs {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }

    .custom-tab {
        display: flex;
        align-items: center;
        flex-direction: column;
        padding: 10px 15px;
        border-radius: 10px;
        background-color: #004aad;
        color: white;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 180px;
    }

    .custom-tab:hover {
        background-color: #0077b6;
        transform: scale(1.05);
    }

    .custom-tab img {
        width: 40px;
        height: 40px;
        margin-bottom: 8px;
    }

    .custom-tab.selected {
        background-color: #0077b6;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Tab Content */
    .content-container {
        text-align: center;
        margin-top: 20px;
    }

    /* Footer Section */
    footer {
        text-align: center;
        font-size: 14px;
        color: #555;
        padding: 10px;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown("""
<div class="header-container">
    <img src="logo/logo.png" class="header-logo" alt="AI-MED Logo">
    <div class="header-text">
        <div class="header-title">Welcome to AI-MED Models UK</div>
        <div class="header-subtitle">Transforming Healthcare with AI</div>
    </div>
    <a href="http://www.aimedmodels.com" target="_blank" class="header-link">🌐 Visit AI-MED Models</a>
</div>
""", unsafe_allow_html=True)

# Tab Navigation
tabs = [
    {"label": "Brain Tumor Detection", "icon": "logo/brain_tumor_icon.png"},
    {"label": "Lung Cancer Detection", "icon": "logo/lung_cancer_icon.png"},
    {"label": "Eye Disease Detection", "icon": "logo/eye_disease_icon.png"},
    {"label": "Heart Disease Detection", "icon": "logo/heart_disease_icon.png"},
    {"label": "Breast Cancer Detection", "icon": "logo/breast_cancer_icon.png"},
]

selected_tab = st.radio(
    "Select a Model",
    options=[tab["label"] for tab in tabs],
    index=0,
    key="tabs_navigation",
    horizontal=True,
)

# Render Content Based on Selected Tab
if selected_tab == "Brain Tumor Detection":
    st.subheader("Brain Tumor Detection")
    image_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(image)
        prediction = brain_tumor_model.predict(img_array)
        predicted_class = brain_tumor_classes[np.argmax(prediction)]
        st.success(f"Prediction: {predicted_class}")

# Lung Cancer Detection Tab
if selected_tab == "Lung Cancer Detection":
    st.subheader("Lung Cancer Detection")
    image_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"], key="lung_cancer_upload")
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(image)
        prediction = lung_cancer_model.predict(img_array)
        predicted_class = lung_cancer_classes[np.argmax(prediction)]
        st.success(f"Prediction: {predicted_class}")

# Eye Disease Detection Tab
if selected_tab == "Eye Disease Detection":
    st.subheader("Eye Disease Detection")
    image_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"], key="eye_disease_upload")
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(image)
        prediction = eye_disease_model.predict(img_array)
        predicted_class = eye_disease_classes[np.argmax(prediction)]
        st.success(f"Prediction: {predicted_class}")

# Heart Disease Detection Tab
if selected_tab == "Heart Disease Detection":
    st.subheader("Heart Disease Detection")
    option = st.radio("Select Input Method", ["Manual Input", "Upload CSV"], key="heart_disease_radio")
    
    if option == "Manual Input":
        # Manual input for heart disease features
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

        # Preprocess input and predict
        heart_data = preprocess_tabular(heart_inputs)
        prediction = heart_disease_model.predict(heart_data)
        predicted_class = heart_disease_classes[int(prediction[0] > 0.5)]

        # Display prediction
        st.success(f"Prediction: {predicted_class}")

    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            required_columns = [
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ]
            if all(col in df.columns for col in required_columns):
                heart_data = df[required_columns].values
                predictions = heart_disease_model.predict(heart_data)
                predicted_classes = [heart_disease_classes[int(pred > 0.5)] for pred in predictions]
                df["Prediction"] = predicted_classes
                st.markdown("<h4>Batch Predictions:</h4>", unsafe_allow_html=True)
                st.dataframe(df)
            else:
                st.error("The uploaded CSV does not have the required columns. Please check your file.")

# Breast Cancer Detection Tab
if selected_tab == "Breast Cancer Detection":
    st.subheader("Breast Cancer Detection")
    option = st.radio("Select Input Method", ["Manual Input", "Upload CSV"], key="breast_cancer_radio")

    if option == "Manual Input":
        # Manual input for all 30 features
        breast_inputs = [
            st.number_input("Radius Mean", value=14.0),
            st.number_input("Texture Mean", value=19.0),
            st.number_input("Perimeter Mean", value=90.0),
            st.number_input("Area Mean", value=600.0),
            st.number_input("Smoothness Mean", value=0.1),
            st.number_input("Compactness Mean", value=0.2),
            st.number_input("Concavity Mean", value=0.3),
            st.number_input("Concave Points Mean", value=0.1),
            st.number_input("Symmetry Mean", value=0.2),
            st.number_input("Fractal Dimension Mean", value=0.07),
            st.number_input("Radius SE", value=0.5),
            st.number_input("Texture SE", value=1.0),
            st.number_input("Perimeter SE", value=3.0),
            st.number_input("Area SE", value=20.0),
            st.number_input("Smoothness SE", value=0.005),
            st.number_input("Compactness SE", value=0.02),
            st.number_input("Concavity SE", value=0.03),
            st.number_input("Concave Points SE", value=0.01),
            st.number_input("Symmetry SE", value=0.02),
            st.number_input("Fractal Dimension SE", value=0.002),
            st.number_input("Radius Worst", value=15.0),
            st.number_input("Texture Worst", value=25.0),
            st.number_input("Perimeter Worst", value=100.0),
            st.number_input("Area Worst", value=800.0),
            st.number_input("Smoothness Worst", value=0.15),
            st.number_input("Compactness Worst", value=0.25),
            st.number_input("Concavity Worst", value=0.35),
            st.number_input("Concave Points Worst", value=0.15),
            st.number_input("Symmetry Worst", value=0.3),
            st.number_input("Fractal Dimension Worst", value=0.08)
        ]

        # Preprocess and predict
        breast_data = preprocess_tabular(breast_inputs, scaler=breast_cancer_scaler)
        prediction = breast_cancer_model.predict(breast_data)
        predicted_class = breast_cancer_classes[int(prediction[0][0] > 0.5)]
        st.success(f"Prediction: {predicted_class}")

    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="breast_cancer_csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] == 30:
                breast_data = breast_cancer_scaler.transform(df)
                predictions = breast_cancer_model.predict(breast_data)
                predicted_classes = [breast_cancer_classes[int(pred[0] > 0.5)] for pred in predictions]
                df["Prediction"] = predicted_classes
                st.markdown("<h4>Batch Predictions:</h4>", unsafe_allow_html=True)
                st.dataframe(df)
            else:
                st.error("The uploaded CSV does not have the required 30 features. Please check your file.")

# Footer
st.markdown(
    """
    <footer>
        &copy; 2024 AI-MED Models UK | Transforming Healthcare with AI
    </footer>
    """,
    unsafe_allow_html=True,
)
