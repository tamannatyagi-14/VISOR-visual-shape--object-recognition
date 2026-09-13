import cv2
from shape_detector.config import *

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #BGR frame → single channel grayscale


def apply_blur(gray):
    return cv2.GaussianBlur(gray, (7, 7), 0)           #Remove noise using Gaussian blur

def apply_threshold(blurred):
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,              # Convert to pure black & white using adaptive threshold
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=4
    )


def preprocess(frame):
    gray    = to_grayscale(frame)
    blurred = apply_blur(gray)
    thresh  = apply_threshold(blurred)
    return thresh

