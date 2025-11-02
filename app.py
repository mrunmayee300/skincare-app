from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import numpy as np

app = Flask(__name__)
model = YOLO("model/best.pt")

# basic skincare advice mapping
SKINCARE_TIPS = {
    "acne": "Use a gentle cleanser with salicylic acid. Avoid touching your face and use non-comedogenic moisturizers.",
    "dark_circle": "Get enough sleep and stay hydrated. Try caffeine or vitamin C-based eye creams.",
    "wrinkle": "Use retinol or peptides at night. Always apply sunscreen to prevent further damage.",
    "dry_skin": "Moisturize twice daily with products containing ceramides or hyaluronic acid.",
    "oily_skin": "Use a lightweight, oil-free moisturizer and gentle exfoliants like BHA once or twice a week."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['image']
    img_path = os.path.join('static/uploads', image.filename)
    image.save(img_path)

    # YOLO prediction
    results = model.predict(source=img_path, save=True, conf=0.4)
    detections = results[0].boxes.cls.tolist()
    names = [model.names[int(cls)] for cls in detections]

    if not names:
        summary = "No visible skin issues detected."
        tips = "Your skin looks healthy. Maintain a consistent skincare routine."
    else:
        summary = ", ".join(set(names)).replace("_", " ").title()
        tips = "\n".join(
            [f"- {SKINCARE_TIPS.get(name, 'No tips available for this condition.')}" for name in set(names)]
        )

    return render_template(
        'result.html',
        image=image.filename,
        summary=summary,
        tips=tips
    )

if __name__ == '__main__':
    app.run(debug=True)
