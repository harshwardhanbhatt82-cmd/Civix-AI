# 🔍 Civix AI
### AI-Powered Road Damage Detection and Complaint Management System

Civix AI is a deep learning-based web application designed to detect road damage and provide an integrated complaint management system.

## 🚀 Features

- AI-based road damage detection using YOLO.
- Detection of potholes, cracks and open manholes.
- Confidence score for detected damage.
- Dashboard with detection statistics.
- Detection history and PDF/CSV reports.
- User registration and login.
- Citizen complaint submission and tracking.
- Admin dashboard for complaint status management.

## 🛠️ Technologies Used

- Python
- Streamlit
- YOLO / Ultralytics
- OpenCV
- SQLite
- Pandas
- ReportLab

## 📊 Model Performance

Validation results after model training:

| Metric | Score |
|---|---:|
| Precision | 78.07% |
| Recall | 68.58% |
| mAP@50 | 73.13% |
| mAP@50-95 | 39.84% |

*These are validation results, not final test-set results.*

## 💻 Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/harshwardhanbhatt82-cmd/Civix-AI.git
```

2. Open the project folder:

```bash
cd Civix-AI
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python -m streamlit run app.py
```
## 🌐 Live Demo

Try Civix AI here:

👉 [Launch Civix AI](https://harshwardhanbhatt82-cmd-civix-ai-app-5fvt9e.streamlit.app/)

## 🎯 Project Objective

The main objective of Civix AI is to combine automated road damage detection with citizen complaint management and reporting in one platform.

## ⚠️ Disclaimer

This is an academic demonstration project. It is not officially connected to any municipal corporation or government authority. Complaints are managed within the demonstration application.

## 👨‍💻 Project Information

**Project Name:** Civix AI

**Domain:** Artificial Intelligence and Deep Learning

**Application Type:** Web Application

**GitHub:** https://github.com/harshwardhanbhatt82-cmd/Civix-AI

## 🔮 Future Scope

- Integration with real-time road monitoring systems.
- GPS-based damage location mapping.
- Integration with official complaint management platforms.
- Improved detection performance using larger datasets.
