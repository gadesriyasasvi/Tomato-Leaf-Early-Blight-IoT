import cv2
import numpy as np
import tensorflow as tf

# 1. Load your Best Model
# Change to 'tomato_custom_cnn.h5' if you want to test that instead
MODEL_PATH = 'tomato_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def predict_on_leaf(image_path):
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not find image: {image_path}")
        return

    # 2. ISOLATE GREEN (Background Removal)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Focus on the largest green object
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Crop and Preprocess
        leaf_crop = frame[y:y+h, x:x+w]
        img = cv2.resize(leaf_crop, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # 3. AI Prediction
        preds = model.predict(img, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]

        # 4. Results Logic (Higher threshold for disease)
        if class_idx == 1 and confidence > 0.90:
            label = "DISEASED"
            color = (0, 0, 255) # Red
        else:
            label = "HEALTHY"
            color = (0, 255, 0) # Green

        # Draw box and label for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        print(f"Result for {image_path}: {label} ({confidence*100:.1f}%)")
        
        # Show the processed crop and the result
        cv2.imshow("What the AI sees (Crop)", leaf_crop)
        cv2.imshow("Final Result", frame)
        cv2.waitKey(0)
    else:
        print(f"❌ No leaf detected in {image_path}")

# Run tests
#predict_on_leaf('healthy_3.jpg')
predict_on_leaf('healthy_4.jpg')