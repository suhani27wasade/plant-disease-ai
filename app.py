from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
import tensorflow as tf

from PIL import Image
from werkzeug.utils import secure_filename

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"
DATABASE = "plant_history.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

input_shape = model.input_shape

IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]

print("====================================")
print("Model loaded successfully!")
print("Number of classes:", len(class_names))
print("Image size:", IMG_WIDTH, "x", IMG_HEIGHT)
print("====================================")


# =========================================================
# DATABASE
# =========================================================

def init_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            disease TEXT,
            confidence REAL,
            severity TEXT,
            symptoms TEXT,
            treatment TEXT,
            prevention TEXT,
            created_at TEXT
        )
    """)

    connection.commit()

    connection.close()


init_database()


# =========================================================
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Tomato___Late_blight": {
        "symptoms":
            "Dark brown or black spots may appear on leaves. "
            "Affected areas can spread quickly, especially under "
            "cool and humid conditions.",

        "treatment":
            "Remove severely infected leaves and plant parts. "
            "Improve air circulation and avoid watering directly "
            "over the leaves.",

        "prevention":
            "Avoid excess moisture on leaves, provide adequate "
            "spacing and remove infected plant debris.",

        "severity":
            "High"
    },


    "Tomato___Early_blight": {
        "symptoms":
            "Brown circular spots with darker rings may develop "
            "on older leaves. Yellowing can occur around infected areas.",

        "treatment":
            "Remove affected leaves and keep the surrounding area clean. "
            "Avoid overhead watering.",

        "prevention":
            "Maintain proper plant spacing and remove infected leaves "
            "and plant debris.",

        "severity":
            "Medium"
    },


    "Tomato___healthy": {
        "symptoms":
            "The leaf appears generally healthy with no strong visible "
            "signs associated with the diseases in the model.",

        "treatment":
            "No specific disease treatment is indicated. Continue "
            "normal plant care.",

        "prevention":
            "Maintain balanced watering, adequate sunlight and "
            "good air circulation.",

        "severity":
            "Low"
    },


    "Potato___Late_blight": {
        "symptoms":
            "Dark irregular lesions may appear on leaves and can "
            "spread rapidly under humid conditions.",

        "treatment":
            "Remove severely affected plant material and avoid "
            "excess moisture on foliage.",

        "prevention":
            "Improve air circulation, avoid prolonged leaf wetness "
            "and remove infected plant debris.",

        "severity":
            "High"
    },


    "Potato___Early_blight": {
        "symptoms":
            "Small dark spots may develop into larger circular "
            "lesions, often with concentric rings.",

        "treatment":
            "Remove affected leaves and maintain proper plant care.",

        "prevention":
            "Use proper spacing, maintain plant nutrition and "
            "remove infected plant debris.",

        "severity":
            "Medium"
    },


    "Potato___healthy": {
        "symptoms":
            "No strong visible symptoms of the diseases represented "
            "in the model were detected.",

        "treatment":
            "No specific disease treatment is required. Continue "
            "normal plant care.",

        "prevention":
            "Maintain healthy soil, balanced watering, sunlight "
            "and regular monitoring.",

        "severity":
            "Low"
    }
}


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return "Please select an image."


    image = request.files["image"]


    if image.filename == "":

        return "Please select an image."


    # -----------------------------------------------------
    # Generate unique filename
    # -----------------------------------------------------

    original_name = secure_filename(
        image.filename
    )

    extension = os.path.splitext(
        original_name
    )[1].lower()


    if extension not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        return "Invalid image format."


    filename = (
        str(uuid.uuid4())
        + extension
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    image.save(filepath)


    try:

        # -------------------------------------------------
        # IMAGE PROCESSING
        # -------------------------------------------------

        img = Image.open(
            filepath
        ).convert("RGB")


        img = img.resize(
            (
                IMG_WIDTH,
                IMG_HEIGHT
            )
        )


        img_array = np.array(
            img
        )


        img_array = (
            img_array / 255.0
        )


        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # -------------------------------------------------
        # AI PREDICTION
        # -------------------------------------------------

        predictions = model.predict(
            img_array,
            verbose=0
        )


        predicted_index = np.argmax(
            predictions[0]
        )


        disease = class_names[
            predicted_index
        ]


        confidence = float(
            predictions[0][
                predicted_index
            ]
        ) * 100


        disease_display = (
            disease
            .replace("___", " - ")
            .replace("_", " ")
        )


        # -------------------------------------------------
        # DISEASE INFO
        # -------------------------------------------------

        info = DISEASE_INFO.get(

            disease,

            {
                "symptoms":
                    "The AI model detected a condition "
                    "associated with this class.",

                "treatment":
                    "Remove severely affected plant parts "
                    "where appropriate and maintain good "
                    "plant-care practices.",

                "prevention":
                    "Maintain proper watering, sunlight, "
                    "air circulation and regular monitoring.",

                "severity":
                    "Medium"
            }
        )


        severity = info[
            "severity"
        ]


        # -------------------------------------------------
        # SAVE TO DATABASE
        # -------------------------------------------------

        connection = sqlite3.connect(
            DATABASE
        )

        cursor = connection.cursor()


        cursor.execute(
            """
            INSERT INTO scans
            (
                filename,
                disease,
                confidence,
                severity,
                symptoms,
                treatment,
                prevention,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                filename,
                disease_display,
                round(confidence, 2),
                severity,
                info["symptoms"],
                info["treatment"],
                info["prevention"],
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )


        scan_id = cursor.lastrowid


        connection.commit()

        connection.close()


        # -------------------------------------------------
        # RESULT PAGE
        # -------------------------------------------------

        return render_template(

            "result.html",

            image=filename,

            disease=disease_display,

            confidence=round(
                confidence,
                2
            ),

            severity=severity,

            symptoms=info[
                "symptoms"
            ],

            treatment=info[
                "treatment"
            ],

            prevention=info[
                "prevention"
            ],

            scan_id=scan_id
        )


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )


        return f"""
        <h2>Prediction Error</h2>

        <p>{str(e)}</p>

        <a href="/">
            Go Back
        </a>
        """


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM scans
        ORDER BY id DESC
        """
    )


    scans = cursor.fetchall()

    connection.close()


    return render_template(
        "history.html",
        scans=scans
    )


# =========================================================
# DELETE ONE HISTORY
# =========================================================

@app.route(
    "/history/delete/<int:scan_id>",
    methods=["POST"]
)
def delete_history(scan_id):

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute(
        "SELECT filename FROM scans WHERE id = ?",
        (scan_id,)
    )


    result = cursor.fetchone()


    if result:

        filename = result[0]

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        if os.path.exists(filepath):

            os.remove(filepath)


    cursor.execute(
        "DELETE FROM scans WHERE id = ?",
        (scan_id,)
    )


    connection.commit()

    connection.close()


    return redirect(
        url_for("history")
    )


# =========================================================
# CLEAR HISTORY
# =========================================================

@app.route(
    "/history/clear",
    methods=["POST"]
)
def clear_history():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute(
        "SELECT filename FROM scans"
    )


    files = cursor.fetchall()


    for file in files:

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file[0]
        )


        if os.path.exists(filepath):

            os.remove(filepath)


    cursor.execute(
        "DELETE FROM scans"
    )


    connection.commit()

    connection.close()


    return redirect(
        url_for("history")
    )


# =========================================================
# PDF REPORT
# =========================================================

@app.route(
    "/download-report/<int:scan_id>"
)
def download_report(scan_id):

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()


    cursor.execute(
        "SELECT * FROM scans WHERE id = ?",
        (scan_id,)
    )


    scan = cursor.fetchone()

    connection.close()


    if not scan:

        return "Report not found."


    pdf_filename = (
        f"plant_disease_report_{scan_id}.pdf"
    )


    pdf_path = os.path.join(
        REPORT_FOLDER,
        pdf_filename
    )


    # -----------------------------------------------------
    # PDF DOCUMENT
    # -----------------------------------------------------

    document = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )


    styles = getSampleStyleSheet()


    title_style = styles[
        "Title"
    ]

    title_style.alignment = (
        TA_CENTER
    )


    heading_style = styles[
        "Heading2"
    ]


    normal_style = styles[
        "BodyText"
    ]


    elements = []


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    elements.append(

        Paragraph(
            "Plant Disease AI",
            title_style
        )

    )


    elements.append(
        Spacer(1, 10)
    )


    elements.append(

        Paragraph(
            "AI Plant Health Diagnosis Report",
            heading_style
        )

    )


    elements.append(
        Spacer(1, 20)
    )


    # -----------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------

    summary_data = [

        [
            "Disease / Class",
            scan["disease"]
        ],

        [
            "Confidence",
            f'{scan["confidence"]}%'
        ],

        [
            "Severity",
            scan["severity"]
        ],

        [
            "Date",
            scan["created_at"]
        ]

    ]


    summary_table = Table(
        summary_data,
        colWidths=[
            130,
            340
        ]
    )


    summary_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0,0),
                (0,-1),
                colors.HexColor(
                    "#E7F5EA"
                )
            ),

            (
                "TEXTCOLOR",
                (0,0),
                (0,-1),
                colors.HexColor(
                    "#126B35"
                )
            ),

            (
                "GRID",
                (0,0),
                (-1,-1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0,0),
                (-1,-1),
                "TOP"
            ),

            (
                "PADDING",
                (0,0),
                (-1,-1),
                8
            )

        ])

    )


    elements.append(
        summary_table
    )


    elements.append(
        Spacer(1, 25)
    )


    # -----------------------------------------------------
    # SYMPTOMS
    # -----------------------------------------------------

    elements.append(

        Paragraph(
            "Symptoms",
            heading_style
        )

    )


    elements.append(
        Paragraph(
            scan["symptoms"],
            normal_style
        )
    )


    elements.append(
        Spacer(1, 18)
    )


    # -----------------------------------------------------
    # TREATMENT
    # -----------------------------------------------------

    elements.append(

        Paragraph(
            "Treatment",
            heading_style
        )

    )


    elements.append(
        Paragraph(
            scan["treatment"],
            normal_style
        )
    )


    elements.append(
        Spacer(1, 18)
    )


    # -----------------------------------------------------
    # PREVENTION
    # -----------------------------------------------------

    elements.append(

        Paragraph(
            "Prevention",
            heading_style
        )

    )


    elements.append(
        Paragraph(
            scan["prevention"],
            normal_style
        )
    )


    elements.append(
        Spacer(1, 25)
    )


    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    elements.append(

        Paragraph(
            "<b>Important:</b> This AI result is an "
            "image-based prediction intended for "
            "informational purposes. For serious crop "
            "problems, confirm the diagnosis with a "
            "qualified agricultural professional.",
            normal_style
        )

    )


    # -----------------------------------------------------
    # BUILD PDF
    # -----------------------------------------------------

    document.build(
        elements
    )


    return send_file(
        pdf_path,
        as_attachment=True
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",port=5000,debug=True
    )