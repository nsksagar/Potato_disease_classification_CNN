# Potato Disease Classification (CNN)

An end-to-end deep learning application that classifies potato leaf diseases from images. A Convolutional Neural Network (CNN) is trained on the [PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) dataset to detect **Early Blight**, **Late Blight**, or classify a leaf as **Healthy**, served through a FastAPI backend and a React frontend for real-time predictions.

## Overview

Potato crops are commonly affected by fungal diseases like Early Blight and Late Blight, which can significantly reduce yield if not detected early. This project automates disease detection from a photo of a potato leaf, giving farmers and agronomists a fast, accessible diagnostic tool.

## How It Works

```
Leaf Image (upload)
        │
        ▼
┌───────────────────┐
│  React Frontend    │  → user uploads/drags an image
└─────────┬──────────┘
          │  POST /predict
          ▼
┌───────────────────┐
│  FastAPI Backend    │  → preprocesses image, calls model
└─────────┬──────────┘
          │  REST call
          ▼
┌───────────────────┐
│ TensorFlow Serving  │  → runs inference on the CNN model
└─────────┬──────────┘
          │
          ▼
   Predicted Class + Confidence Score
   (Early Blight / Late Blight / Healthy)
```

## Tech Stack

- **Model:** Convolutional Neural Network (CNN) — TensorFlow / Keras
- **Serving:** TensorFlow Serving
- **Backend API:** FastAPI + Uvicorn
- **Frontend:** React + Material-UI
- **Dataset:** PlantVillage (potato subset — Early Blight, Late Blight, Healthy)

## Model Architecture & Training

The model is a sequential CNN trained on 256x256 RGB images:

- Resizing & rescaling layer, followed by data augmentation (random flip/rotation) to reduce overfitting
- 5 convolutional blocks (`Conv2D` + `MaxPooling2D`), with filter sizes 32 → 64 → 64 → 64 → 64
- Dense layers leading to a 3-class softmax output

**Training configuration:**

| Parameter | Value |
|---|---|
| Image size | 256 x 256 |
| Batch size | 32 |
| Epochs | 50 |
| Classes | Early Blight, Late Blight, Healthy |

**Result:** ~97% validation accuracy after training, with training/validation accuracy and loss curves plotted in the training notebook.

Training code and experiments live in [`Training/training.ipynb`](Training/training.ipynb).

## Project Structure

```
Potato_disease_classification_CNN/
├── Training/
│   ├── training.ipynb       # Model training, evaluation, and plots
│   └── PlantVillage/        # Dataset (Early Blight, Late Blight, Healthy)
├── api/
│   ├── main.py               # FastAPI inference endpoint
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── home.js           # Image upload + prediction UI
│   └── package.json
├── models/
│   └── 1.keras                # Trained model
├── convert.py                 # Converts .keras model to TF SavedModel format
├── model.config                # TensorFlow Serving model config
└── potatoes.h5                 # Trained model (HDF5 format)
```

## Setup & Usage

### 1. Train the model (optional — a trained model is already included)

Run [`Training/training.ipynb`](Training/training.ipynb) to retrain the CNN on the PlantVillage dataset.

### 2. Convert the model for serving

```bash
python convert.py
```

This exports `models/1.keras` to TensorFlow's SavedModel format at `models/1`.

### 3. Run TensorFlow Serving

```bash
docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/models:/models" \
    -e MODEL_CONFIG_FILE=/models/model.config \
    tensorflow/serving
```

### 4. Run the FastAPI backend

```bash
cd api
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`, with a `/predict` endpoint that accepts an image file and returns the predicted class and confidence score.

### 5. Run the React frontend

```bash
cd frontend
npm install
npm start
```

Set `REACT_APP_API_URL` in `frontend/.env` to point to your backend's `/predict` endpoint.

## API Reference

**`POST /predict`**

Request: multipart form-data with an image file (`file`).

Response:
```json
{
  "class": "Early Blight",
  "confidence": 0.97
}
```

## Classes

| Label | Description |
|---|---|
| Early Blight | Fungal disease caused by *Alternaria solani* |
| Late Blight | Fungal disease caused by *Phytophthora infestans* |
| Healthy | No disease detected |

## Future Improvements

- Expand to additional crops and disease classes
- Add model versioning and A/B testing via TensorFlow Serving
- Deploy the full stack (frontend, API, model server) to the cloud
- Add unit tests for the API and model preprocessing pipeline
