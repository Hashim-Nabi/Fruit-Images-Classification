# 🍎 Fruit Classification using CNN

A deep learning project that classifies images of fruits into **10 distinct categories** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## 🧠 Problem Statement

Automated fruit classification has practical applications in agriculture, food processing, and retail. Given an image of a fruit, the model predicts which of the 10 fruit classes it belongs to.

---

## 📦 Dataset

- **Source:** [Fruit Classification 10 Class — Kaggle](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)
- **Classes:** 10 types of fruits
- **Split:** 80% Training / 20% Validation

---

## 🏗️ Model Architecture

A custom CNN with 3 convolutional blocks followed by a fully connected classifier:

| Layer | Details |
|---|---|
| Conv Block 1 | Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25) |
| Conv Block 2 | Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25) |
| Conv Block 3 | Conv2D(128) → BatchNorm → MaxPool → Dropout(0.40) |
| Dense Head | Dense(256) → BatchNorm → Dropout(0.50) → Softmax |

- **Input size:** 64×64 RGB images
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy

---

## 📊 Visualizations

The notebook generates 8 graphs:

1. Class distribution bar chart
2. Sample images from each class
3. Pixel intensity histogram
4. Dataset split pie chart
5. Training accuracy curve
6. Training loss curve
7. Confusion matrix
8. Per-class precision & recall

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd fruit-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API

- Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
- Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`

### 4. Run the notebook
```bash
jupyter notebook fruit_classification.ipynb
```

The notebook will automatically download and extract the dataset, train the model, and display all results.

---

## 📦 Requirements

```
tensorflow
scikit-learn
matplotlib
seaborn
numpy
pillow
kaggle
jupyter
```

---

## 📁 Project Structure

```
fruit-classification/
│
├── fruit_classification.ipynb   # Main notebook
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── data/                        # Downloaded dataset (auto-created)
└── fruit_classifier_cnn.h5      # Saved model (after training)
```

---

## 📈 Results

After training, the notebook prints a full evaluation summary including:
- Overall test accuracy
- Weighted precision & recall
- Best and hardest classified fruit class
- Full classification report

---

## 🔮 Future Improvements

- Transfer learning with MobileNetV2 or ResNet50
- Higher resolution inputs (128×128 or 224×224)
- Larger and more diverse dataset
- Real-time fruit classification via webcam

---

## 📄 License

This project is for educational purposes as part of a university semester project.
