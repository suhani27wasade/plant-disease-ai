# AI-Based Plant Disease Detection and Treatment Recommendation System

A web-based machine learning application that detects plant diseases from leaf images and provides treatment recommendations.

## Project Structure

```
plant-disease-ai/
│
├── app.py              # Main Flask application entry point
├── requirements.txt    # Project dependencies and libraries
├── README.md           # Project documentation and setup guide
│
├── dataset/            # PlantVillage dataset directory
│   ├── raw/            # Place unzipped raw PlantVillage dataset here
│   ├── train/          # 70% split for model training
│   ├── validation/     # 15% split for hyperparameter tuning & evaluation
│   └── test/           # 15% split for final test evaluation
│
├── model/              # Stored trained Machine Learning models (e.g. .h5 or .keras files)
├── training/           # Model training scripts and dataset processing code
│   └── prepare_dataset.py # Script to split raw dataset into train/validation/test
│
├── static/             # Static web assets
│   ├── css/            # Custom stylesheets for the web UI
│   ├── js/             # JavaScript files for interactive web features
│   └── uploads/        # Temporary directory for uploaded leaf images
│
├── templates/          # HTML templates for rendering web pages
│
└── utils/              # Helper modules (image processing, recommendation engine logic)
```

---

## Dataset Setup Instructions (PlantVillage)

### 1. Download PlantVillage Dataset
Download the **PlantVillage** dataset from Kaggle or GitHub (e.g., [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emware/plantvillage-dataset)).

### 2. Place Original Dataset in `dataset/raw/`
Extract the downloaded zip archive so that all disease class folders (e.g., `Tomato___Bacterial_spot`, `Apple___Black_rot`, etc.) are placed inside `dataset/raw/`:

```
plant-disease-ai/dataset/raw/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Tomato___Bacterial_spot/
└── ...
```

### 3. Run Dataset Preparation Script
Open your terminal, navigate to the `plant-disease-ai` folder, and execute:

```bash
python training/prepare_dataset.py
```

This script will:
- Automatically scan all class folders inside `dataset/raw/`.
- Shuffle and split images randomly into **70% Train**, **15% Validation**, and **15% Test**.
- Preserve original class folder names and full image resolutions.
- Prevent duplicate images across splits.
- Print detailed progress and summary counts for each class.

---

## Running the Application (Local Development)

```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.
