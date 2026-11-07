import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask_green = cv2.inRange(hsv, np.array([25, 40, 40]),  np.array([85, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([5,  30, 30]),  np.array([30, 255, 200]))
    mask       = cv2.bitwise_or(mask_green, mask_brown)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Show what's being detected
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = frame.copy()
    frame_area = frame.shape[0] * frame.shape[1]
    
    for c in contours:
        area = cv2.contourArea(c)
        pct  = area / frame_area * 100
        x, y, w, h = cv2.boundingRect(c)
        color = (0, 255, 0) if pct >= 8 else (0, 0, 255)  # green = detected, red = too small
        cv2.rectangle(debug, (x, y), (x+w, y+h), color, 2)
        cv2.putText(debug, f"{pct:.1f}%", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Original", debug)
    cv2.imshow("Mask (white = detected)", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()