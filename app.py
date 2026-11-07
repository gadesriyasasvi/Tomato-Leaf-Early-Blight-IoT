
import cv2
import numpy as np
import tensorflow as tf

# 1. Load your trained model
# Ensure the filename matches exactly what you downloaded from Colab
model_path = 'tomato_model_2class.h5' 
model = tf.keras.models.load_model(model_path)

# 2. Start the Webcam
cap = cv2.VideoCapture(0)

print("AI System Starting... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Pre-process the image for the AI
    # Resize to 128x128 and normalize (same as your training code)
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 4. Run Prediction
    predictions = model.predict(img, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    # 5. Logic for Spraying (Hardware Simulation)
    # Assuming Index 0 = Healthy, Index 1 = Early_blight
    if class_idx == 1:
        if confidence > 0.85:
            status = "SEVERE DISEASE"
            color = (0, 0, 255)  # Red
            action = "SPRAY: 5 Seconds"
        else:
            status = "MILD DISEASE"
            color = (0, 255, 255) # Yellow
            action = "SPRAY: 2 Seconds"
    else:
        status = "HEALTHY"
        color = (0, 255, 0)   # Green
        action = "NO SPRAY"

    # 6. Display results on the Webcam Feed
    display_text = f"{status} ({confidence*100:.1f}%)"
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, action, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Smart Pesticide Sprayer AI", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()