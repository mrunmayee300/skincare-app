import cv2
import numpy as np

# Use a pre-shipped face detector (Haar). You can swap to DNN or mediapipe for better results.
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_and_crop(image_bgr, target_size=(640,640)):
    """Detect largest face, crop and return resized RGB image (target_size).
    image_bgr: numpy array BGR (cv2 default)
    returns: cropped_rgb (H,W,3) or None if no face
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None
    # choose largest face
    x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    # expand box slightly for cheeks/forehead
    pad = int(0.2 * max(w,h))
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(image_bgr.shape[1], x+w+pad)
    y2 = min(image_bgr.shape[0], y+h+pad)
    crop = image_bgr[y1:y2, x1:x2]
    resized = cv2.resize(crop, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb
