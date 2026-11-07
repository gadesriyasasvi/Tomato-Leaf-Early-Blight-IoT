"""
Smart Pesticide Sprayer — Live Webcam Classifier
=================================================
Uses the trained EfficientNetB3 model to classify tomato leaves
from a webcam feed and tells the ESP8266 how long to spray.

Severity is determined by GradCAM (% leaf area affected),
NOT by confidence score — confidence only controls uncertainty.

ESP8266 communication: sends a single character over Serial
  'N' → No spray   (Healthy)
  'M' → Mild spray (Early Blight, <30% area)
  'S' → Severe spray (Early Blight, ≥30% area)
  'U' → Uncertain, no spray
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import serial          # pip install pyserial
import time
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────
MODEL_PATH           = 'tomato_model.keras'   # updated from .h5
IMG_SIZE             = 300                     # must match training
CONFIDENCE_THRESHOLD = 0.65
SEVERITY_THRESHOLD   = 0.30                   # 30% leaf area
GRADCAM_THRESHOLD    = 0.45

# Class names in the exact alphabetical order Keras assigned during training
# Check what train.py printed at the end — should be:
CLASS_NAMES = ["Tomato___Early_blight", "Tomato___healthy"]
#               index 0                   index 1

# ESP8266 Serial port — change to your port
# Windows: 'COM3', 'COM4' etc.
# Linux/Mac: '/dev/ttyUSB0' or '/dev/tty.usbserial-...'
SERIAL_PORT    = 'COM3'
SERIAL_BAUD    = 9600
USE_SERIAL     = False   # set True when ESP8266 is connected

# How many seconds between predictions (avoid spamming the model)
PREDICTION_INTERVAL = 2.0
# ──────────────────────────────────────────────────────────────


# ── GRADCAM ──────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, pred_index, model):
    img_tensor = tf.Variable(tf.cast(img_array, tf.float32))

    with tf.GradientTape() as tape:
        preds = model(img_tensor, training=False)
        score = preds[:, pred_index]

    grads   = tape.gradient(score, img_tensor)
    heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def segment_leaf(image_bgr):
    hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([25, 20, 20]),  np.array([100, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([5,  30, 30]),  np.array([30,  255, 200]))
    leaf_mask  = cv2.bitwise_or(mask_green, mask_brown)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    leaf_mask  = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask  = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN,  kernel)
    return leaf_mask


def get_severity(heatmap, frame_bgr):
    h, w           = frame_bgr.shape[:2]
    hmap_resized   = cv2.resize(heatmap, (w, h))
    activated      = (hmap_resized >= GRADCAM_THRESHOLD).astype(np.uint8) * 255
    leaf_mask      = segment_leaf(frame_bgr)
    leaf_px        = np.sum(leaf_mask > 0)
    diseased_px    = np.sum((activated > 0) & (leaf_mask > 0))
    if leaf_px == 0:
        leaf_px     = h * w
        diseased_px = np.sum(activated > 0)
    return diseased_px / leaf_px

def detect_and_crop_leaf(frame):
    """
    Detects the largest leaf-like object in the frame.
    Returns the cropped leaf region, or None if no leaf found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Green (healthy tissue) + yellow-brown (blight spots)
    mask_green = cv2.inRange(hsv, np.array([25, 40, 40]),  np.array([85, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([5,  30, 30]),  np.array([30, 255, 200]))
    mask       = cv2.bitwise_or(mask_green, mask_brown)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    # Ignore if too small — probably noise, not a leaf
    # Minimum 8% of frame area
    frame_area = frame.shape[0] * frame.shape[1]
    if area < frame_area * 0.08:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)

    # Add padding around the crop
    pad = 20
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(frame.shape[1], x + w + pad)
    y2  = min(frame.shape[0], y + h + pad)

    crop = frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2, y2)
    return crop, bbox

# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("🔄 Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.\n")

    # Connect to ESP8266
    ser = None
    if USE_SERIAL:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            time.sleep(2)  # wait for ESP8266 to boot
            print(f"✅ Serial connected on {SERIAL_PORT}")
        except Exception as e:
            print(f"⚠️  Serial connection failed: {e}")
            print("   Running without hardware.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Webcam started. Press 'q' to quit.\n")

    last_predict_time = 0
    current_status    = "Initializing..."
    current_color     = (200, 200, 200)
    current_action    = ""
    serial_cmd        = 'U'

    VOTE_BUFFER      = []
    VOTES_NEEDED     = 5
    VOTE_THRESHOLD   = 0.6
    last_sent_cmd    = None
    GRADCAM_OVERRIDE_THRESHOLD = 0.01

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # Only run prediction every PREDICTION_INTERVAL seconds
        if now - last_predict_time >= PREDICTION_INTERVAL:
            last_predict_time = now

            # Preprocess
            leaf_crop, bbox = detect_and_crop_leaf(frame)

            if leaf_crop is None:
                # No leaf detected — skip prediction, show waiting message
                current_status = "No leaf detected"
                current_color  = (200, 200, 200)
                current_action = "Hold a leaf up to the camera"
                serial_cmd     = 'U'
                # Draw the overlay and continue without predicting
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, current_status, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, current_color, 2)
                cv2.putText(frame, current_action, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Smart Pesticide Sprayer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Draw green box around detected leaf
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Leaf detected", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Use the cropped leaf for prediction instead of full frame
            resized = cv2.resize(leaf_crop, (IMG_SIZE, IMG_SIZE))
            arr     = np.expand_dims(resized.astype(np.float32), 0)
            # Note: NO manual /255 normalization — EfficientNetB3 handles it internally

            preds      = model.predict(arr, verbose=0)[0]
            pred_idx   = int(np.argmax(preds))
            confidence = float(preds[pred_idx])
            label      = CLASS_NAMES[pred_idx]
            is_blight  = "blight" in label.lower() or "early" in label.lower()

            if confidence < CONFIDENCE_THRESHOLD:
                current_status = "UNCERTAIN"
                current_color  = (200, 200, 200)
                current_action = "No spray — check manually"
                serial_cmd     = 'U'

            elif not is_blight:
                current_status = "HEALTHY"
                current_color  = (0, 200, 0)
                current_action = "No spray needed"
                serial_cmd     = 'N'

            else:
                # GradCAM severity
                heatmap = make_gradcam_heatmap(arr, pred_idx, model)
                severity = get_severity(heatmap, resized)

                if severity >= SEVERITY_THRESHOLD:
                    current_status = "EARLY BLIGHT — SEVERE"
                    current_color  = (0, 0, 255)
                    current_action = f"SPRAY: 5 sec | {severity*100:.1f}% affected"
                    serial_cmd     = 'S'
                else:
                    current_status = "EARLY BLIGHT — MILD"
                    current_color  = (0, 200, 255)
                    current_action = f"SPRAY: 2 sec | {severity*100:.1f}% affected"
                    serial_cmd     = 'M'
            # Send command to ESP8266
            if ser and ser.is_open:
                ser.write(serial_cmd.encode())
                print(f"📡 Sent to ESP8266: '{serial_cmd}' — {current_status}")

        # ── Draw overlay on frame ─────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, current_status,
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, current_color, 2)
        cv2.putText(frame, current_action,
                    (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {confidence*100:.1f}%" if 'confidence' in dir() else "",
                    (15, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Smart Pesticide Sprayer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()