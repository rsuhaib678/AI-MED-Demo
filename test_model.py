from tensorflow.keras.models import load_model

# Update paths to your models
models_to_reload = [
    ("models/lung_cancer_model.h5", "models/lung_cancer_model_v2_7_0.h5"),
    ("models/eye_disease_model.h5", "models/eye_disease_model_v2_7_0.h5"),
    ("models/brain_tumor_model.h5", "models/brain_tumor_model_v2_7_0.h5"),
    ("models/breast_cancer_model.h5", "models/breast_cancer_model_v2_7_0.h5"),
]

for original, updated in models_to_reload:
    model = load_model(original, compile=False)  # Load with the current TensorFlow version
    model.save(updated)  # Save in a compatible format
    print(f"Model {original} saved as {updated}")
