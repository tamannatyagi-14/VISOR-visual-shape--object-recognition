# VISOR — Visual Shape & Object Recognition System

<div align="center">

![VISOR](https://img.shields.io/badge/VISOR-v1.0-gold?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![PyQt5](https://img.shields.io/badge/PyQt5-5.x-red?style=for-the-badge)
![DRDO](https://img.shields.io/badge/DRDO-Internship-navy?style=for-the-badge)

**Real-time Shape & Color Detection System**

*Developed during 6-week internship at ADRDE – DRDO, Ministry of Defence, Government of India*

</div>

---

## 📌 About

**VISOR (Visual Shape & Object Recognition)** is a real-time computer vision system that detects and recognizes a predefined custom shape through a standard webcam anywhere within the camera frame.

The system uses classical image processing techniques — **no deep learning, no GPU required** — making it lightweight and efficient on standard hardware.

> 🏆 Rated **"Very Good"** by mentor Mr. Ajitabh Tiwari, ADRDE-DRDO

---

## 🎥 Demo

> *Detection in action — Gold bounding box around detected shape with real-time statistics*

![VISOR Demo](assets/demo.jpeg)

*(Add your demo GIF here — record screen while VISOR detects your shape)*

---

## ✨ Features

- 🔍 **Real-time shape detection** — detects custom shape anywhere in webcam frame
- 🎨 **HSV-based color verification** — validates both shape geometry AND color
- 📊 **Live statistics** — Height, Width, Area, Perimeter, Confidence Score
- 🖥️ **PyQt5 GUI** — Black Gold theme with interactive controls
- ▶️ **Start / Stop / Reset** — fully interactive buttons
- 📈 **Session tracking** — Count, Last detection, Best confidence
- ⚡ **30 FPS real-time** — runs on standard laptop webcam

---

## 🏗️ System Architecture

```
Webcam Input
     ↓
Image Preprocessing
(Grayscale → Gaussian Blur → Adaptive Threshold)
     ↓
Shape Detection
(Contour Detection → Hu Moment Matching)
     ↓
Color Verification
(BGR → HSV → Color Masking)
     ↓
Decision Engine
(Shape + Color both match?)
     ↓
Result Visualization
(Bounding Box + Statistics on PyQt5 UI)
```

---

## 📁 Project Structure

```
VISOR/
├── shape_detector/
│   ├── __init__.py          # Package initializer
│   ├── config.py            # All settings & thresholds
│   ├── preprocessor.py      # Frame cleaning pipeline
│   ├── detector.py          # Shape + color detection
│   └── utils.py             # Dimensions + confidence
├── assets/                  # Template images
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_detector.py
├── main.py                  # Entry point + PyQt5 UI
├── requirements.txt         # Dependencies
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| OpenCV | 4.x | Image processing & detection |
| NumPy | 1.x | Array operations |
| PyQt5 | 5.x | GUI development |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/VISOR-Shape-Detection.git
cd VISOR-Shape-Detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run VISOR**
```bash
python main.py
```

---

## 🎯 How It Works

1. **Print or draw** your target shape (nested squares with red, gold & white colors)
2. **Click START** in the VISOR interface
3. **Hold the shape** in front of your webcam
4. VISOR detects the shape and shows a **gold bounding box** with live statistics!

---

## 📊 Detection Parameters

You can tune detection in `shape_detector/config.py`:

```python
MATCH_THRESHOLD = 0.45    # Lower = stricter match
MIN_AREA = 200            # Minimum shape size in pixels
```

---

## 🔬 Algorithms Used

| Algorithm | Purpose |
|---|---|
| Adaptive Thresholding | Handles variable lighting conditions |
| `cv2.findContours()` with RETR_TREE | Extracts nested shape hierarchy |
| Hu Moment Matching | Shape comparison independent of size & rotation |
| HSV Color Masking | Color verification independent of lighting |

---

## 🚀 Future Scope

- [ ] YOLOv8 integration for deep learning-based detection
- [ ] Multiple shape support simultaneously
- [ ] Deployment on Raspberry Pi
- [ ] Sound alert on detection

---

## 🏢 About the Internship

This project was developed during a **6-week internship** at:

**ADRDE – Aerial Delivery Research and Development Establishment**
**DRDO — Defence Research and Development Organisation**
Ministry of Defence, Government of India
Station Road, Agra Cantt.

- **Intern:** Tamanna Tyagi
- **Mentor:** Mr. Ajitabh Tiwari
- **Period:** June 2026 – July 2026
- **Performance:** Very Good

---

## 👩‍💻 Developer

**Tamanna Tyagi**
B.Tech CSE — Raja Balwant Singh Engineering Technical Campus, Agra


![LinkedIn][![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/tamanna-tyagi-1409200527t)


---

<div align="center">
Made with ❤️ during DRDO Internship 2026
</div>