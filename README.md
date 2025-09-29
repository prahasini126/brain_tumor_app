### 🧠 Brain Tumor Detection App

A machine learning project for real-time brain tumor detection using MRI scans and a Convolutional Neural Network (CNN) model, deployed as an interactive web application with Streamlit.

### 📌 Features

- Upload MRI brain scans.

- AI predicts whether a brain tumor is present.

- Displays prediction probability.

- Clean, user-friendly UI with Streamlit.

### 🛠 Tech Stack

- Python

- TensorFlow / Keras — CNN model

- Streamlit — Web interface

- OpenCV — Image preprocessing

- Pillow — Image handling

- NumPy — Array operations

### 🚀 Setup Instructions

- Clone or Download the Repository

- git clone <your-repo-url>

- cd brain_tumor_app

### Install Dependencies

- pip install streamlit tensorflow opencv-python pillow numpy


### Place the Model

- Ensure brain_tumor_cnn_model.h5 is inside the project folder.

### Run the App

- streamlit run app.py

### 🖥 Usage

- Open the app in your browser (usually http://localhost:8501).

- Upload an MRI image (JPG/PNG).

- View the prediction and probability score.

### 📌 Model Details

- The model is a CNN trained on MRI brain scan images labeled for tumor presence.

- It uses image preprocessing, convolution layers, pooling layers, and dense layers to make predictions.

### ⚡ Future Improvements

- Deploy online (Streamlit Cloud or Hugging Face Spaces).

- Add multi-class classification for tumor types.

- Integrate a dataset uploader for live model retraining.

### 📜 License

This project is for educational purposes and is open-source.