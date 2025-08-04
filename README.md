# Vehicle Classifier API and UI

An end-to-end MLOps pipeline for vehicle classification using CNN, Flask API, and Streamlit UI with continuous retraining capabilities.

## 🚗 About

This project classifies images as either "Car" or "Motorcycle" using a Convolutional Neural Network. Built with MLOps best practices, it features a Flask REST API for model serving and a Streamlit web interface for easy interaction.

### ✨ Features

- 🧠 **CNN Model**: TensorFlow-based vehicle classifier
- 🔄 **Auto Retraining**: API endpoint for model updates with new data
- 🌐 **REST API**: Flask-based prediction service
- 💻 **Web UI**: Streamlit interface for predictions and monitoring
- 🚀 **Auto Deploy**: Infrastructure as Code with render.yaml
- 📊 **Load Testing**: Performance evaluation with Locust
- 📈 **Monitoring**: Real-time API health and metrics

## 📁 Project Structure

```
vehicle-classifier/
├── 🔧 api/
│   └── app.py                     # Flask API server
├── 📦 src/
│   ├── data_loader.py             # Data preprocessing
│   ├── model_builder.py           # CNN architecture
│   ├── predictor.py               # Prediction logic
│   └── train.py                   # Training pipeline
├── 🖥️ streamlit_app/
│   └── streamlit_app.py           # Web interface
├── 📊 data/
│   ├── train/
│   │   ├── Car/                   # Car training images
│   │   └── Motorcycle/            # Motorcycle training images
│   └── test/
│       ├── Car/                   # Car test images
│       └── Motorcycle/            # Motorcycle test images
├── 🤖 models/
│   └── vehicle_classifier_model.keras # Trained model
├── ⚙️ render.yaml                 # Deployment config
├── 📋 requirements.txt            # Dependencies
└── 📖 README.md                   # You are here
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager
- Render.com account (for deployment)

### Installation

**1. Clone the repo**
```bash
git clone https://github.com/Gakwaya011/Summative-assignment---MLOP.git
cd vehicle-classifier
```

**2. Set up environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

### Run Locally

**Start the API** 🔧
```bash
export FLASK_APP=api/app.py  # Windows: set FLASK_APP=api/app.py
flask run
```
→ API runs at `http://127.0.0.1:5000`

**Start the UI** 💻
```bash
streamlit run streamlit_app/streamlit_app.py
```
→ UI runs at `http://localhost:8501`

> 💡 **Tip**: Update the API URL in `streamlit_app.py` for local testing

## 🛠️ API Reference

### Endpoints

#### `GET /` - Health Check
Check if API is running
```bash
curl http://localhost:5000/
```

#### `POST /predict` - Classify Image
Upload an image and get vehicle classification

**Form Upload:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

**JSON Base64:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_string"}' \
  http://localhost:5000/predict
```

**Response:**
```json
{
  "predicted_class": "Car",
  "confidence": 0.98
}
```

#### `POST /retrain` - Update Model
Upload new training data to retrain the model

```bash
curl -X POST -F "zip_file=@training_data.zip" http://localhost:5000/retrain
```

**Requirements:**
- ZIP file with `Car/` and `Motorcycle/` folders
- Images in respective folders

**Response:**
```json
{
  "message": "Retraining completed successfully.",
  "metrics": {
    "accuracy": 0.95,
    "val_accuracy": 0.92
  }
}
```

## 🚀 Deployment

Deploy to [Render.com](https://render.com) with zero configuration using `render.yaml`:

- **🔧 API Service**: Flask app with Gunicorn
https://vehicle-classifier-api.onrender.com/

- **💻 UI Service**: Streamlit web interface
https://vehicle-classifier-ui.onrender.com/

Just connect your GitHub repo and Render handles the rest!

## 📺 Demo

[🎥 Watch the Demo](https://youtu.be/WIwkCnmefeU) - See predictions and retraining in action

## ⚡ Performance Testing

Load tested with **Locust** to ensure production readiness:

| Configuration | Max Users | RPS | Median Latency | 95th Percentile |
|---------------|-----------|-----|----------------|-----------------|
| 1 Container   | 50      | 43 | 22 ms         | 420 ms          |
| 2 Containers  | 100      | 45 | 180 ms         | 300 ms          |



## 🏗️ MLOps Best Practices

This project showcases production-ready MLOps patterns:

- ✅ **Model Optimization**: L2 regularization + data augmentation
- ✅ **Efficient Data Pipeline**: TensorFlow `tf.data` with caching/prefetching  
- ✅ **Continuous Training**: Zero-downtime model updates via API
- ✅ **Stateless API**: Horizontally scalable design
- ✅ **Infrastructure as Code**: Version-controlled deployment
- ✅ **Monitoring**: Real-time health checks and metrics
- ✅ **Load Testing**: Performance validation under stress

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Model** | TensorFlow, CNN |
| **API** | Flask, Gunicorn |
| **Frontend** | Streamlit |
| **Deployment** | Render.com |
| **Testing** | Locust |





**Project Link** - [https://github.com/Gakwaya011/Summative-assignment---MLOP.git](https://github.com/Gakwaya011/Summative-assignment---MLOP.git)

---

⭐ **Star this repo** if you found it helpful!
