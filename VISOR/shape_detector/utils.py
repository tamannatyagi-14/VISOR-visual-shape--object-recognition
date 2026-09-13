import cv2
import numpy as np
from shape_detector.config import *

def get_dimensions(contour):
    x, y, w, h = cv2.boundingRect(contour)
    area        = cv2.contourArea(contour)
    perimeter   = cv2.arcLength(contour, True)
    
    return {
        "height":    h,
        "width":     w,
        "area":      round(area),
        "perimeter": round(perimeter)
    }

def calculate_confidence(score):
    if score is None:
        return 0
    
    confidence = max(0, 1 - (score / MATCH_THRESHOLD))
    return round(confidence * 100)

def get_center(contour):
    M = cv2.moments(contour)
    
    if M["m00"] == 0:
        return None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return cx, cy
