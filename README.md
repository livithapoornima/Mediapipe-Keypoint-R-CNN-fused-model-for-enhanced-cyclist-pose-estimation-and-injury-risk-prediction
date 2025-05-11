# Fusion Pose: A Mediapipe and Keypoint R-CNN Fused Model for Enhanced Cyclist Pose Estimation and Injury Risk Prediction

This project implements a fusion-based pose estimation system using **MediaPipe** and **Keypoint R-CNN** to improve the accuracy of cyclist posture analysis. It calculates joint angles and predicts injury risk levels using machine learning models like XGBoost.

---

## 🔍 Overview

Cyclists often experience posture-related injuries due to incorrect riding positions. This project enhances injury risk prediction by fusing outputs from multiple pose estimation models and analyzing joint-level movement patterns.

The pipeline includes:
- Keypoint detection (multiple DL models)
- Feature fusion
- Joint angle computation (hip, knee, elbow)
- Injury risk classification
- Validation using ground truth and multiple performance metrics

---

## 🧠 Models & Techniques

- **Pose Estimators**:
  - MediaPipe
  - Keypoint R-CNN (TorchVision)
  - YOLOPose
  - MoveNet

- **Fusion Model**: Confidence-based merging of keypoints from different detectors

- **Risk Prediction**: XGBoost classifier trained on angle features

- **Validation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Percentage of Correct Keypoints (PCK)
  - Mean Joint Position Error (MJPE)
  - Euclidean Distance

---

## ⚙️ Technologies Used

- Python
- Torch / TorchVision
- OpenCV
- MediaPipe
- XGBoost
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebooks

---

## 🛠️ Key Features

- Extracts 2D keypoints from cycling videos using DL models
- Combines outputs of MediaPipe and Keypoint R-CNN for enhanced accuracy
- Calculates biomechanical joint angles
- Predicts injury risk level (Low / Moderate / High)
- Compares results with marker-based ground truth
- Evaluates performance using industry-standard metrics

---

## 📂 Project Structure

fusion-pose/
├── notebooks/
│ ├── angles.ipynb
│ ├── euclidean_distance.ipynb
│ ├── featurefusion_updated.ipynb
│ ├── groundtruth.ipynb
│ ├── mae.ipynb
│ ├── mediapip.ipynb
│ ├── mjpe.ipynb
│ ├── movenet.ipynb
│ ├── mseAndRmse.ipynb
│ ├── pck.ipynb
│ ├── risk_classification.ipynb
│ ├── validation.ipynb
│ ├── video_to_frame.ipynb
│ └── xgboost_classifier.ipynb
├── utils/ # Helper scripts
├── sample_videos/ # Input test videos
├── requirements.txt
└── README.md


##  Getting Started

### 1. Clone the repository
git clone https://github.com/livithapoornima/Mediapipe-Keypoint-R-CNN-fused-model-for-enhanced-cyclist-pose-estimation-and-injury-risk-prediction.git
cd Mediapipe-Keypoint-R-CNN-fused-model-for-enhanced-cyclist-pose-estimation-and-injury-risk-prediction


### 2. Install dependencies

pip install -r requirements.txt


---

## 📊 Evaluation Metrics

- **MAE / MSE / RMSE** – Quantitative comparison between predicted and ground-truth joint angles.
- **MJPE (Mean Joint Position Error)** – Measures average error per joint keypoint.
- **PCK (Percentage of Correct Keypoints)** – Evaluates how many keypoints fall within a threshold distance from ground truth.
- **Euclidean Distance** – Used for keypoint-level comparison between models and fused output.
- **Confusion Matrix / Accuracy** – For evaluating injury risk classification.

---

## 📦 Dataset

The dataset includes:
- Video frames of cyclists in different postures
- Marker-based ground-truth joint annotations
- Processed keypoint data from MediaPipe, Keypoint R-CNN, YOLOPose, and MoveNet

⚠️ Due to size restrictions, the dataset is not stored in this repository.  

---
## 🙏 Acknowledgements

- **Google MediaPipe** – for fast real-time pose estimation
- **TorchVision** – for Keypoint R-CNN model
- **MoveNet / YOLOPose** – for alternative pose extraction
- **VIT Chennai – Cyclist Posture Lab** – for research support and ground-truth data
- Open-source ML and CV communities for continued inspiration

---
