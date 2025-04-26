# 🛠️ Automated Product Defect Detection using YOLOv8

This project implements a real-time **defect detection system** for metallic tins and cans using **YOLOv8 Nano** and a user-friendly **Flask web interface**. It aims to automate quality control processes in manufacturing, improving accuracy and efficiency while reducing human error and inspection time.

## 🔍 Project Overview

Defect detection in metallic containers like tins is crucial for maintaining product quality in industries such as food, chemical, and cosmetics packaging. Traditional manual inspection is slow, inconsistent, and labor-intensive. This project proposes a smart solution using deep learning (YOLOv8) and computer vision to detect:
- Scratches
- Dents
- Surface deformities
- Irregular patterns

## ⚙️ Features

- 🔎 Real-time defect detection using YOLOv8 Nano
- 🖼️ Upload images of tins via a web interface
- 🧠 Support for both **pre-trained** and **custom-trained** models
- 📊 Display detection results with bounding boxes and confidence scores
- 📈 Visual analytics including accuracy, precision, recall, and F1-score
- 📁 Easy dataset management and model training interface
- 🧪 Evaluation using confusion matrix and testing scenarios

## 🖥️ System Architecture

Image Upload → Image Preprocessing → YOLOv8 Detection → Result Display (Web UI)


## 💡 Technologies Used

- Python 3.10+
- Flask (Web framework)
- Ultralytics YOLOv8
- OpenCV
- Pandas, Numpy, Matplotlib
- HTML/CSS (Flask Templates)
- Bootstrap (Optional UI enhancements)

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/defect-detection-yolov8.git
   cd defect-detection-yolov8
pip install -r requirements.txt

🛠️ Future Enhancements

    Real-time integration with conveyor belt and industrial cameras

    Feedback system for adaptive model retraining

    Multi-defect classification

    Support for mobile/web-based uploads
