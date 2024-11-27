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

def preprocess_csv(csv_data, scaler=None):
    preprocessed = scaler.transform(csv_data) if scaler else csv_data
    return preprocessed

# Streamlit App Layout
st.set_page_config(page_title="AI-MED Models UK", page_icon="🩺")

st.markdown(
    """
    <style>
        .header-title {
            text-align: center;
            color: #1A73E8;
            font-size: 48px;
            margin: 20px 0;
        }
        .sub-title {
            text-align: center;
            color: #555555;
            font-size: 24px;
        }
        .logo-style {
            margin-left: 20px;
        }
        .prediction {
            color: #1A73E8;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .uploaded-csv {
            background-color: #f4f4f9;
            padding: 10px;
            border-radius: 10px;
            font-size: 14px;
        }
        .tabs > div[data-baseweb="tab"] {
            font-size: 20px !important;
        }
    </style>
    """, unsafe_allow_html=True
)

# Header Layout
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.image("logo/logo.png", width=100, use_column_width=False)  # Removed 'className'

with col2:
    st.markdown(
        """
        <div style="text-align: center; font-size: 30px; font-weight: bold; line-height: 1.5;">
            Welcome to<br>
            <span style="color: #007BFF;">AI-MED Models UK</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.write("")  # Leave this empty to maintain layout

st.markdown('<div class="sub-title">Transforming Healthcare with AI</div>', unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'>"
    "<a href='http://www.aimedmodels.com' style='color: #1A73E8;'>Visit AI-MED Models</a>"
    "</h3>",
    unsafe_allow_html=True
)

# Tabs for Models
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 **Brain Tumor Detection**",
    "🫁 **Lung Cancer Detection**",
    "👁️ **Eye Disease Detection**",
    "💗 **Heart Disease Detection**",
    "🎗️ **Breast Cancer Detection**"
])

# Brain Tumor Tab
with tab1:
    st.subheader("Brain Tumor Detection")
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(img)
        prediction = brain_tumor_model.predict(img_array)
        result = brain_tumor_classes[np.argmax(prediction)]
        st.markdown(f"<div class='prediction'>Prediction: {result}</div>", unsafe_allow_html=True)

# Lung Cancer Tab
with tab2:
    st.subheader("Lung Cancer Detection")
    uploaded_file = st.file_uploader("Upload Lung X-Ray Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(img)
        prediction = lung_cancer_model.predict(img_array)
        result = lung_cancer_classes[np.argmax(prediction)]
        st.markdown(f"<div class='prediction'>Prediction: {result}</div>", unsafe_allow_html=True)

# Eye Disease Tab
with tab3:
    st.subheader("Eye Disease Detection")
    uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(img)
        prediction = eye_disease_model.predict(img_array)
        result = eye_disease_classes[np.argmax(prediction)]
        st.markdown(f"<div class='prediction'>Prediction: {result}</div>", unsafe_allow_html=True)

# Heart Disease Tab
with tab4:
    st.subheader("Heart Disease Detection")
    option = st.radio("Select Input Method", options=["Manual Input", "Upload CSV"])
    if option == "Manual Input":
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
            result = heart_disease_classes[int(prediction[0] > 0.5)]
            st.markdown(f"<div class='prediction'>Prediction: {result}</div>", unsafe_allow_html=True)
    elif option == "Upload CSV":
        uploaded_csv = st.file_uploader("Upload CSV for Bulk Prediction", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.write("Uploaded CSV:")
            st.dataframe(df, className="uploaded-csv")
            predictions = heart_disease_model.predict(preprocess_csv(df.values))
            results = [heart_disease_classes[int(pred[0] > 0.5)] for pred in predictions]
            df["Prediction"] = results
            st.write("Predictions:")
            st.dataframe(df)

# Breast Cancer Tab
with tab5:
    st.subheader("Breast Cancer Detection")
    option = st.radio("Select Input Method", options=["Manual Input", "Upload CSV"])
    if option == "Manual Input":
        # Default values for all 30 features
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

        if st.button("Predict Breast Cancer"):
            breast_data = preprocess_tabular(breast_inputs, scaler=breast_cancer_scaler)
            prediction = breast_cancer_model.predict(breast_data)
            result = breast_cancer_classes[int(prediction[0][0] > 0.5)]
            st.markdown(f"<div class='prediction'>Prediction: {result}</div>", unsafe_allow_html=True)

    elif option == "Upload CSV":
        uploaded_csv = st.file_uploader("Upload CSV for Bulk Prediction", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.write("Uploaded CSV:")
            st.dataframe(df, className="uploaded-csv")
            preprocessed = preprocess_csv(df.values, scaler=breast_cancer_scaler)
            predictions = breast_cancer_model.predict(preprocessed)
            results = [breast_cancer_classes[int(pred[0] > 0.5)] for pred in predictions]
            df["Prediction"] = results
            st.write("Predictions:")
            st.dataframe(df)

# Add Styling
st.markdown("""
    <style>
        .stApp {background-color: #f4f4f9;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
        .stButton>button:hover {background-color: #45a049;}
        .uploaded-csv {
            background-color: #f4f4f9;
            padding: 10px;
            border-radius: 10px;
            font-size: 14px;
        }
        .prediction {
            color: #1A73E8;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .tabs > div[data-baseweb="tab"] {
            font-size: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)
