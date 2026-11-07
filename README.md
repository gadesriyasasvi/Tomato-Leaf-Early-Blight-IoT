# Tomato Leaf Early Blight Detection and IoT Smart Pesticide Sprayer

## Project Overview

This project presents an AI-based tomato leaf disease detection system integrated with an IoT-based smart pesticide sprayer. It uses deep learning and computer vision techniques to identify Early Blight disease in tomato leaves through a webcam.

Based on the detection result, the system determines the spraying action and communicates with an ESP8266 microcontroller.

## Objectives

- Detect Early Blight disease in tomato leaves.
- Classify tomato leaves into Healthy and Early Blight categories.
- Capture and process leaf images using a webcam.
- Estimate disease severity using Grad-CAM.
- Communicate detection results to an ESP8266.
- Automate pesticide spraying based on disease severity.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Deep Learning
- Grad-CAM
- ESP8266
- Serial Communication

## System Architecture

1. Webcam captures the tomato leaf image.
2. OpenCV detects and extracts the leaf region.
3. The trained deep learning model predicts the leaf condition.
4. Grad-CAM estimates the affected leaf area.
5. The system determines the disease severity.
6. A command is communicated to the ESP8266.
7. The microcontroller controls the pesticide spraying mechanism.

## Disease Detection and Spraying Logic

| Command | Condition | Action |
|---|---|---|
| N | Healthy | No spraying |
| M | Mild Early Blight | 2-second spray |
| S | Severe Early Blight | 5-second spray |
| U | Uncertain | No automatic spraying |

## Project Structure

```text
IOT_project/
│
├── main.py
├── app.py
├── app_safety.py
├── check_app.py
├── TomatoTesting/
│   ├── Trained Models
│   ├── Test Images
│   └── Simulation Scripts
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Tomato-Leaf-Early-Blight-IoT.git
```

Navigate into the project directory:

```bash
cd Tomato-Leaf-Early-Blight-IoT
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Run the main application:

```bash
python main.py
```

Press `q` to close the webcam window.

## Hardware Components

- ESP8266 NodeMCU
- Relay module
- DC water pump
- Power supply
- Connecting wires

## Future Enhancements

- Improve disease detection accuracy.
- Support additional tomato leaf diseases.
- Integrate real-time IoT monitoring.
- Develop a mobile application for remote monitoring.
- Improve disease severity estimation.

## License

This project is developed for academic and educational purposes.