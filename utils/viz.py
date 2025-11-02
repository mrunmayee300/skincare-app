import cv2

def draw_boxes_on_bgr(bgr, detections):
    for d in detections:
        cv2.rectangle(bgr, (d['xmin'],d['ymin']), (d['xmax'],d['ymax']), (0,255,0), 2)
        txt = f"{d['class']} {d['conf']:.2f}"
        cv2.putText(bgr, txt, (d['xmin'], d['ymin']-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return bgr

def severity_from_count(detections):
    """Simple rule-based severity:
       0: none, 1: mild (1-5), 2: moderate (6-15), 3: severe(>15)
    """
    n = len(detections)
    if n == 0: return {"score": 0, "label":"No visible lesions"}
    if n <= 5: return {"score": 1, "label":"Mild"}
    if n <= 15: return {"score": 2, "label":"Moderate"}
    return {"score": 3, "label":"Severe"}
