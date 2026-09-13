import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

h, w = frame.shape[:2]
cx, cy = w//2, h//2
region = hsv[cy-50:cy+50, cx-50:cx+50]

pixels = region.reshape(-1, 3)
print("HSV min:", pixels.min(axis=0))
print("HSV max:", pixels.max(axis=0))

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()