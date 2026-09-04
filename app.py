import os
import uuid
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"


print("=" * 60)
print("LOADING PLANT DISEASE AI MODEL...")
print("=" * 60)


# Check model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


# Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]


print("Number of classes:", len(class_names))


# ==========================================
# VALIDATE MODEL OUTPUT
# ==========================================

model_output_classes = model.output_shape[-1]

print("Model output classes:", model_output_classes)


if len(class_names) != model_output_classes:

    raise ValueError(
        f"""
ERROR!

Model has {model_output_classes} output classes
but class_names.txt has {len(class_names)} classes.

Please make sure model and class_names.txt
are generated from the SAME training process.
"""
    )


# ==========================================
# GET IMAGE SIZE
# ==========================================

input_shape = model.input_shape

IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]


print("Image size:", IMG_WIDTH, "x", IMG_HEIGHT)

print("\nCLASS ORDER:")

for index, name in enumerate(class_names):
    print(index, "->", name)

print("=" * 60)


# ==========================================
# ALLOWED FILES
# ==========================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# DISEASE INFORMATION
# ==========================================

def get_disease_info(disease):

    info = {

        "Apple___Apple_scab": {
            "plant": "Apple",
            "diagnosis": "Apple Scab",
            "symptoms": "Dark olive or brown spots may appear on leaves and fruit.",
            "treatment": "Remove infected leaves and maintain good air circulation."
        },

        "Apple___Black_rot": {
            "plant": "Apple",
            "diagnosis": "Black Rot",
            "symptoms": "Dark lesions and rotting areas may develop.",
            "treatment": "Remove infected plant parts and maintain orchard hygiene."
        },

        "Apple___Cedar_apple_rust": {
            "plant": "Apple",
            "diagnosis": "Cedar Apple Rust",
            "symptoms": "Yellow or orange spots may appear on leaves.",
            "treatment": "Remove infected leaves and monitor nearby host plants."
        },

        "Apple___healthy": {
            "plant": "Apple",
            "diagnosis": "Healthy",
            "symptoms": "No significant disease symptoms detected.",
            "treatment": "Continue proper watering and regular plant care."
        },

        "Tomato___Bacterial_spot": {
            "plant": "Tomato",
            "diagnosis": "Bacterial Spot",
            "symptoms": "Small dark spots may appear on leaves.",
            "treatment": "Remove affected leaves and avoid overhead watering."
        },

        "Tomato___Early_blight": {
            "plant": "Tomato",
            "diagnosis": "Early Blight",
            "symptoms": "Dark circular spots with concentric rings.",
            "treatment": "Remove infected leaves and improve air circulation."
        },

        "Tomato___Late_blight": {
            "plant": "Tomato",
            "diagnosis": "Late Blight",
            "symptoms": "Dark brown lesions may spread quickly across leaves.",
            "treatment": "Remove affected leaves and consult an agricultural expert."
        },

        "Tomato___healthy": {
            "plant": "Tomato",
            "diagnosis": "Healthy",
            "symptoms": "No major disease symptoms detected.",
            "treatment": "Continue normal plant care."
        }

    }


    # Exact information available
    if disease in info:
        return info[disease]


    # Default information for all other classes

    parts = disease.split("___")


    plant = parts[0].replace(
        "_",
        " "
    )


    if len(parts) > 1:

        diagnosis = parts[1].replace(
            "_",
            " "
        )

    else:

        diagnosis = disease.replace(
            "_",
            " "
        )


    return {

        "plant": plant,

        "diagnosis": diagnosis,

        "symptoms":
        "Check the plant for unusual spots, discoloration, yellowing or leaf damage.",

        "treatment":
        "Remove severely affected leaves, maintain plant hygiene and consult an agricultural expert if necessary."

    }


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def prepare_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image = image.resize(
        (IMG_WIDTH, IMG_HEIGHT)
    )


    image_array = np.array(
        image,
        dtype=np.float32
    )


    # IMPORTANT:
    # Do NOT divide by 255 here if your model
    # already contains Rescaling(1./255)


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------
        # CHECK IMAGE
        # ----------------------------------

        if "image" not in request.files:

            return (
                "Error: No image uploaded",
                400
            )


        file = request.files["image"]


        if file.filename == "":

            return (
                "Error: Please select an image",
                400
            )


        if not allowed_file(
            file.filename
        ):

            return (
                "Error: Invalid image format",
                400
            )


        # ----------------------------------
        # SAVE IMAGE
        # ----------------------------------

        original_filename = secure_filename(
            file.filename
        )


        unique_filename = (
            str(uuid.uuid4())
            + "_"
            + original_filename
        )


        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )


        file.save(
            image_path
        )


        # ----------------------------------
        # PREPARE IMAGE
        # ----------------------------------

        img_array = prepare_image(
            image_path
        )


        # ----------------------------------
        # MODEL PREDICTION
        # ----------------------------------

        predictions = model.predict(
            img_array,
            verbose=0
        )


        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        confidence = float(
            predictions[0][predicted_index]
        ) * 100


        disease = class_names[
            predicted_index
        ]


        # ----------------------------------
        # TOP 5 PREDICTIONS
        # ----------------------------------

        top_indices = np.argsort(
            predictions[0]
        )[-5:][::-1]


        top_predictions = []


        for i in top_indices:

            top_predictions.append({

                "disease":
                class_names[int(i)],

                "confidence":
                round(
                    float(predictions[0][i]) * 100,
                    2
                )

            })


        # ----------------------------------
        # GET DISEASE INFORMATION
        # ----------------------------------

        info = get_disease_info(
            disease
        )


        # ----------------------------------
        # DEBUG TERMINAL OUTPUT
        # ----------------------------------

        print("\n" + "=" * 60)

        print("PREDICTION RESULT")

        print("=" * 60)


        print(
            "Uploaded image:",
            unique_filename
        )


        print(
            "Predicted index:",
            predicted_index
        )


        print(
            "Predicted disease:",
            disease
        )


        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )


        print("\nTOP 5 PREDICTIONS:")


        for prediction in top_predictions:

            print(

                prediction["disease"],

                "=>",

                prediction["confidence"],

                "%"

            )


        print("=" * 60 + "\n")


        # ----------------------------------
        # RESULT PAGE
        # ----------------------------------

        return render_template(

            "result.html",

            image=unique_filename,

            disease=disease,

            confidence=round(
                confidence,
                2
            ),

            plant=info["plant"],

            diagnosis=info["diagnosis"],

            symptoms=info["symptoms"],

            treatment=info["treatment"],

            top_predictions=top_predictions

        )


    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )


        return (

            "Prediction Error: "
            + str(e),

            500

        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )