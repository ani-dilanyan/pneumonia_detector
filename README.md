# **Pneumonia X-Ray Detector**

This project is a deep learning-powered web application that detects **Pneumonia from chest X-ray images** using a Convolutional Neural Network (CNN).
Users can upload an X-ray image through a Flask interface and receive:

* **Prediction:** Normal or Pneumonia
* **Confidence Score**
* **Displayed uploaded image**

# **Dataset**

The dataset used in this project can be downloaded from Kaggle with this url:

**[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**

## Download Instructions

1. Download the dataset from Kaggle
2. Extract the ZIP file
3. You will obtain a folder named **`chest_xray`**
4. Place the entire `chest_xray/` folder inside your project directory


# Installation & Setup

## 1. Clone or download the project

```
git clone https://github.com/ani-dilanyan/pneumonia_detector
cd pneumonia_detector-main
```

## 2. Create and activate a virtual environment

```
conda create -n tfenv python=3.10
conda activate tfenv
```

## 3. Install dependencies

```
pip install -r requirements.txt
```


# Model Training

Before running the web application, you must train the neural network.

Make sure the `chest_xray/` folder is inside the project, then run:

```
python train_pneumonia_model.py
```

This will:

* Load the dataset
* Train a CNN
* Save the trained model as:

```
app/models/pneumonia_cnn.h5
```


# Running the Flask Web App

After training the model, start the application:

```
python run.py
```

Open the application url in a browser. 

# Project Structure

The final structure should include:

```
pneumonia_detector/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── models/
│   │   └── pneumonia_cnn.h5 (generated after training)
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── chest_xray/ (dataset from Kaggle)
│   ├── train/
│   ├── val/
│   ├── test/             
│
├── train_pneumonia_model.py
├── run.py
├── config.py
├── requirements.txt
└── README.md
```

# Technologies Used

* **Python 3.10**
* **TensorFlow / Keras**
* **Flask**
* **NumPy**
* **Pillow**
* **Bootstrap 5**


# Notes

* This project is for **educational and learning purposes only**
* The model should **not be used for clinical diagnosis**
