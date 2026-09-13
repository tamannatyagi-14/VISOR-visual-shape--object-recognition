import cv2
import numpy as np
from shape_detector.config import *

def find_contours(thresh):
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy

def match_shape(contour, template):
    score = cv2.matchShapes(
        template,
        contour,
        cv2.CONTOURS_MATCH_I1,
        0.0
    )
    return score

def check_color(frame, contour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for color_name, (lower, upper) in TARGET_COLORS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask  = cv2.inRange(hsv, lower, upper)
        mask_total = cv2.bitwise_or(mask_total, mask)
    
    shape_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(shape_mask, [contour], -1, 255, -1)
    
    result  = cv2.bitwise_and(mask_total, shape_mask)
    total   = cv2.countNonZero(shape_mask)
    colored = cv2.countNonZero(result)
    
    if total == 0:
        return False
    
    return (colored / total) > 0.15

def detect(frame, thresh, template):
    contours, hierarchy = find_contours(thresh)
    
    best_match   = None
    best_score   = float("inf")
    best_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        
        score = match_shape(contour, template)
        
        if score < best_score:
            best_score   = score
            best_contour = contour
    
    if best_score < MATCH_THRESHOLD:
        color_ok = check_color(frame, best_contour)
        if color_ok:
            return best_contour, best_score
    
    return None, None