import cv2
import numpy as np
from insightface.app import FaceAnalysis
from functools import wraps
from ultralytics import YOLO
from flask import Flask, request, jsonify
from db import db, Employee, process_checkinout, Attendance
from sklearn.metrics.pairwise import cosine_similarity


ENROLL_SIM_THRESHOLD = 0.65  # stricter than recognition (0.45)


spoof_model = YOLO("spoof.pt")  # auto-loads torch internally


app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


def extract_embedding_from_image(img):
    faces = app.get(img)
    if len(faces) != 1:
        return None
    return faces[0].embedding


def generate_embedding_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emb = extract_embedding_from_image(frame)
        if emb is not None:
            embeddings.append(emb)

    cap.release()

    if len(embeddings) < 30:
        return None

    return np.mean(embeddings, axis=0)
def detect_spoof(frame):
    """
    Returns True  → REAL face
    Returns False → SPOOF / unsafe
    """

    results = spoof_model(frame, conf=0.5, verbose=False)

    # No detection = unsafe
    if not results or len(results[0].boxes) == 0:
        return False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])

        if cls_id == 0:   # ⛔ spoof detected
            return False

    # At least one REAL face and no spoof
    return True


# ==============================
# ROLE based Access
# ==============================
def role_required(*roles):

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            emp_id = request.headers.get("Emp-Id") #<------ admin / hr deteils
            Passward = request.headers.get("Passward")
            if not emp_id or not Passward:
                return jsonify({"success": False, "message": "Authentication required"}), 401

            emp = Employee.query.filter_by(emp_id=emp_id, Passward=Passward).first()
            if not emp:
                return jsonify({"success": False, "message": "Invalid credentials"}), 401

            if emp.role not in roles:
                return jsonify({"success": False, "message": "Permission denied"}), 403

            # Add emp object to kwargs if needed
            return f(*args, **kwargs)
        return decorated_function
    return decorator
    
def is_duplicate_face(new_embedding):
    """
    Returns (True, emp_id, similarity) if duplicate found
    Else returns (False, None, None)
    """
    employees = Employee.query.filter(
        Employee.embedding.isnot(None)
    ).all()

    for emp in employees:
        score = cosine_similarity(
            new_embedding.reshape(1, -1),
            emp.embedding.reshape(1, -1)
        )[0][0]

        if score >= ENROLL_SIM_THRESHOLD:
            return True, emp.emp_id, round(float(score), 3)

    return False, None, None
