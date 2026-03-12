# 🌿 MEDI — Medicinal Plant Identification Using Deep Learning

An intelligent web application that automatically identifies **40 medicinal plant species** from images using Convolutional Neural Networks (CNN), MobileNet, and a hybrid MobileNet+RNN model.

> 🎓 Academic Final Year Project — B.Tech Computer Science, Amrita Vishwa Vidyapeetham (2025)
> 👤 **Katta Rahul Krishna**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Models & Results](#models--results)
- [Tech Stack](#tech-stack)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## 🔍 Overview

Traditional medicinal plant identification relies heavily on botanical expertise and is time-consuming, impractical for large-scale use, and inaccessible in remote or field environments. **MEDI** solves this by automating the identification process using state-of-the-art deep learning techniques.

The system trains and evaluates three models to classify 40 distinct medicinal plant species:

| Model | Validation Accuracy |
|---|---|
| CNN | 80.13% |
| MobileNet | 90.74% |
| **Hybrid MobileNet + RNN** | **92.94% ✅ Best** |

---

## ✨ Features

- 🔐 User registration and secure login
- 📤 Image upload for real-time plant identification
- 🤖 Three deep learning models for classification
- 📊 Confidence scores and plant information on results
- 📱 Optimized for resource-constrained environments via MobileNet
- 🗃️ MySQL-backed user and data management

---

## 🧠 Models & Results

### CNN (Convolutional Neural Network)
- Architecture: Stacked convolutional layers with 3×3 filters, ReLU activations, and max pooling
- Training: 50 epochs, batch size 32, Adam optimizer, early stopping
- Excelled at: Ashoka, Betel Nut, Bamboo, Castor
- Struggled with: Curry Leaf, Guava, Neem, Pepper

### MobileNet
- Architecture: Depthwise separable convolutions with ImageNet transfer learning
- Training: Same setup as CNN with fine-tuned dense output layer (40 units, softmax)
- Excelled at: Aloevera, Amala, Avocado, Ashwagandha
- Struggled with: Betel, Jasmine

### Hybrid MobileNet + RNN (Best Model)
- Architecture: MobileNet feature extractor → Global Average Pooling → LSTM → Dense (40 units, softmax)
- Combines spatial feature extraction (MobileNet) with contextual/sequential modeling (LSTM)
- Best performer across: Ashoka, Avocado, Bamboo, Betel Nut, Doddapatre, Hibiscus, Papaya, and more

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Web Framework | Flask |
| Deep Learning | TensorFlow, PyTorch |
| Data Processing | Pandas |
| Frontend | HTML, CSS, Bootstrap, JavaScript |
| Database | MySQL |
| IDE | VS Code |
| Server | XAMPP |
| OS | Windows 7 / 8 / 10 |

---

## 💻 System Requirements

### Hardware
- Processor: Intel i3 or higher
- RAM: 8 GB minimum
- Storage: 128 GB

### Software
- Python 3.x
- MySQL
- XAMPP Server
- VS Code (recommended)

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kattaRahulkrishna/medi-plant-identification.git
   cd medi-plant-identification
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the database**
   - Start XAMPP and launch MySQL
   - Create the database schema:
     ```bash
     mysql -u root -p < database/schema.sql
     ```
   - Update DB credentials in `config.py`

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the app**
   Open your browser and navigate to `http://localhost:5000`

---

## 📖 Usage

1. **Register** a new account on the Registration page
2. **Login** with your credentials
3. Navigate to the **Upload** page
4. **Select and upload** an image of a medicinal plant
5. Click **Submit** — the model analyzes the image and returns:
   - Identified plant species
   - Description and medicinal uses
   - Confidence score
6. If the image is unrecognized, the system will prompt you to try a clearer image

---

## 📁 Project Structure

```
medi-plant-identification/
│
├── app.py                    # Main Flask application
├── config.py                 # Configuration (DB, secret key)
├── requirements.txt
│
├── models/
│   ├── cnn_model.py          # CNN model definition and training
│   ├── mobilenet_model.py    # MobileNet model definition and training
│   └── hybrid_model.py       # MobileNet + RNN hybrid model
│
├── dataset/
│   ├── train/                # Training images (80%)
│   ├── val/                  # Validation images (10%)
│   └── test/                 # Test images (10%)
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/              # User-uploaded images
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── register.html
│   ├── login.html
│   ├── home.html
│   ├── upload.html
│   └── result.html
│
└── database/
    └── schema.sql
```

---

## 🗂️ Dataset

- **40 medicinal plant species** included
- Images captured under varied lighting, angles, and backgrounds
- Preprocessing:
  - Resized to **224×224 pixels**
  - Normalized to **[0, 1]** range
  - Augmented via rotations, flips, scaling, and color adjustments
- Split: **80% train / 10% validation / 10% test**

---

## 🧪 Testing

The system was validated through multiple testing strategies:

| Test Type | Result |
|---|---|
| Unit Testing | ✅ Passed |
| Integration Testing | ✅ Passed |
| Functional Testing | ✅ Passed |
| Acceptance Testing | ✅ Passed |
| Black Box Testing | ✅ Passed |
| White Box Testing | ✅ Passed |

All 7 model-building test cases passed successfully, covering dataset loading, image preprocessing, feature extraction, model integration, training, evaluation, and classification output.

---

## 🔮 Future Enhancements

- Integration of **transformer-based** or **attention mechanism** models
- **Synthetic data generation** to address class imbalances
- Incorporation of **botanical metadata** (leaf texture, petal count) for richer classification
- **Mobile app** deployment for field use
- Continuous learning pipeline for real-world adaptation

---

## 📚 References

- Hsu, C.-C., & Tsai, C.-F. (2022). *Deep Learning for Plant Disease Classification.* Computers and Electronics in Agriculture.
- Jiang, Y., & Zhang, H. (2022). *Efficient Plant Classification Using CNNs.* IEEE Access.
- Wu, Z., et al. (2022). *Survey on Deep Learning for Plant Disease Diagnosis.* Computers and Electronics in Agriculture.
- Tan, M., & Le, Q. V. (2021). *EfficientNetV2.* CVPR.
- Liu, Z., et al. (2023). *Swin Transformer V2.* CVPR.

---

## 👨‍💻 Author

**Katta Rahul Krishna**
- 📧 kattarahulkrishna05853@gmail.com
---

## 📄 License

This project was developed for academic purposes (final year project) as part of the B.Tech Computer Science curriculum at **Amrita Vishwa Vidyapeetham, Coimbatore**.
