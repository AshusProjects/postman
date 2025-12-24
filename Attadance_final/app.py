# app.py
from flask import Flask, request, jsonify
from functools import wraps
from db import db, Employee, process_checkinout, Attendance
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from face_engine import generate_embedding_from_video, extract_embedding_from_image, detect_spoof,role_required, is_duplicate_face  # your face functions
from datetime import datetime
from ultralytics import YOLO


SPOOF_THRESHOLD = 0.5  # adjust if needed

ENROLL_SIM_THRESHOLD = 0.65  # stricter than recognition (0.45)



SIM_THRESHOLD = 0.45  # similarity threshold

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

    # Create default admin only if 1001 does not exist
    admin = Employee.query.filter_by(emp_id="1001").first()
    if not admin:
        admin = Employee(
            emp_id="1001",
            name="Admin",
            email_id="admin@gmail.com",
            Passward="admin123",
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()
        print("Default admin created!")
    else:
        print("Admin already exists")


##---------------------------------------------------------------------------
##---------------------------------------------------------------------------

@app.route("/api/enroll_employee", methods=["POST"])
@role_required('admin', 'hr')
def enroll_employee():
    emp_id = request.form.get("emp_id")
    name = request.form.get("name")
    email = request.form.get("email")
    Passward = request.form.get("password")
    role = request.form.get("role")
    video = request.files.get("video")

    if not all([emp_id, name, email, Passward, role, video]):
        return {"isSuccess": False, "message": "All fields required"}, 400

    # 1️⃣ Check duplicate emp_id
    if Employee.query.filter_by(emp_id=emp_id).first():
        return {
            "isSuccess": False,
            "message": "Employee ID already exists"
        }, 400

    # 2️⃣ Save video temporarily
    temp_path = f"temp_{emp_id}.mp4"

    try:
        video.save(temp_path)

        # 3️⃣ Extract embedding
        embedding_vector = generate_embedding_from_video(temp_path)
        if embedding_vector is None:
            return {
                "isSuccess": False,
                "message": "Face embedding could not be generated"
            }, 400

        # 4️⃣ DUPLICATE FACE CHECK 🔒
        is_dup, existing_emp_id, sim = is_duplicate_face(embedding_vector)
        if is_dup:
            return {
                "isSuccess": False,
                "message": f"Face already enrolled as employee {existing_emp_id}",
                "similarity": sim
            }, 409

        # 5️⃣ Save employee
        new_emp = Employee(
            emp_id=emp_id,
            name=name,
            email_id=email,
            Passward=Passward,
            role=role,
            embedding=embedding_vector
        )

        db.session.add(new_emp)
        db.session.commit()

        return {
            "isSuccess": True,
            "message": "Employee enrolled successfully",
            "emp_id": emp_id
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/api/attadance_marking", methods=["POST"])
def recognize_employee():
    if "image" not in request.files:
        return {"isSuccess": False, "message": "Image required"}, 400

    file = request.files["image"]
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # =============================
    # 1️⃣ SPOOF CHECK (YOLO)
    # =============================
    if not detect_spoof(frame):
        return {
            "isSuccess": False,
            "message": "Spoof attack detected"
        }, 403

    # =============================
    # 2️⃣ FACE EMBEDDING
    # =============================
    emb = extract_embedding_from_image(frame)
    if emb is None:
        return {
            "isSuccess": False,
            "message": "No face or multiple faces"
        }, 400

    # =============================
    # 3️⃣ FACE MATCHING
    # =============================
    all_employees = Employee.query.filter(
        Employee.embedding.isnot(None)
    ).all()

    best_emp = None
    best_score = -1

    for emp in all_employees:
        score = cosine_similarity(
            emb.reshape(1, -1),
            emp.embedding.reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_emp = emp

    # =============================
    # 4️⃣ ATTENDANCE
    # =============================
    if best_score >= SIM_THRESHOLD:
        message = process_checkinout(best_emp.emp_id)
        return {
            "isSuccess": True,
            "emp_id": best_emp.emp_id,
            "emp_name": best_emp.name,
            "similarity": round(float(best_score), 3),
            "message": message
        }

    return {
        "isSuccess": False,
        "message": "Unknown person",
        "similarity": round(float(best_score), 3)
    }


@app.route("/api/employees", methods=["GET"])
@role_required('admin', 'hr')
def get_all_employees():
    employees = Employee.query.all()
    emp_list = []

    for emp in employees:
        emp_list.append({
            "emp_id": emp.emp_id,
            "name": emp.name,
            "email": emp.email_id,
            "role": emp.role
        })

    return {"isSuccess": True, "employees": emp_list}
    
@app.route("/api/update_employee/<emp_id>", methods=["PUT"])
@role_required('admin', 'hr')
def update_employee(emp_id):
    emp = Employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        return {"isSuccess": False, "message": "Employee not found"}, 404

    # Get data from request
    name = request.form.get("name")
    email = request.form.get("email")
    Passward = request.form.get("Passward")
    role = request.form.get("role")
    video = request.files.get("video")  # optional new video for embedding

    # Update fields if provided
    if name:
        emp.name = name
    if email:
        emp.email_id = email
    if Passward:
        emp.Passward = Passward
    if role:
        emp.role = role

    # Update embedding if a new video is provided
    if video:
        temp_path = f"temp_update_{emp_id}.mp4"
        video.save(temp_path)
        emp.embedding = generate_embedding_from_video(temp_path)
        os.remove(temp_path)

    db.session.commit()

    return {"isSuccess": True, "message": "Employee updated successfully", "emp_id": emp.emp_id}

@app.route("/api/dashboard", methods=["GET"])
@role_required('admin', 'hr')
def dashboard():
    today = datetime.now().date()
    
    # All employees
    all_employees = Employee.query.all()
    total_employees = len(all_employees)
    
    # Today's attendance records
    today_records = Attendance.query.filter_by(date=today).all()
    
    # Map employee ID -> list of records today
    emp_records_map = {}
    for rec in today_records:
        emp_records_map.setdefault(rec.emp_id, []).append(rec)

    # Calculate present and absent
    present_emp_ids = [emp_id for emp_id, records in emp_records_map.items() if any(r.check_in is not None for r in records)]
    total_present = len(present_emp_ids)
    
    # Absent employees are those with no check-in records
    total_absent = total_employees - total_present

    # Today's attendance details
    attendance_details = []
    for emp in all_employees:
        records = emp_records_map.get(emp.emp_id, [])
        # Take latest record for display
        latest_record = records[-1] if records else None
        attendance_details.append({
            "emp_id": emp.emp_id,
            "name": emp.name,
            "role": emp.role,
            "check_in": latest_record.check_in.strftime("%H:%M:%S") if latest_record and latest_record.check_in else None,
            "check_out": latest_record.check_out.strftime("%H:%M:%S") if latest_record and latest_record.check_out else None,
            "late_by": latest_record.late_by if latest_record else None,
            "early_by": latest_record.early_by if latest_record else None
        })

    return {
        "isSuccess": True,
        "total_employees": total_employees,
        "present": total_present,
        "absent": total_absent,
        "attendance_today": attendance_details
    }

from datetime import datetime, timedelta, date

@app.route("/api/employee_analysis/<emp_id>", methods=["GET"])
@role_required('admin', 'hr')
def employee_analysis(emp_id):
    # Optional query parameters for date range
    start_date_str = request.args.get("start_date")  # YYYY-MM-DD
    end_date_str = request.args.get("end_date")      # YYYY-MM-DD

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str else date.min
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else date.today()

    emp = Employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        return {"isSuccess": False, "message": "Employee not found"}, 404

    # Fetch attendance records for the period
    records = Attendance.query.filter(
        Attendance.emp_id == emp_id,
        Attendance.date >= start_date,
        Attendance.date <= end_date
    ).all()

    if not records:
        return {"isSuccess": True, "employee": emp.name, "message": "No attendance records in this period", "data": {}}

    total_days = (end_date - start_date).days + 1
    days_present = len(set([r.date for r in records if r.check_in]))
    days_absent = total_days - days_present

    # Average working hours
    total_seconds_worked = 0
    checkins_per_day = {}
    total_late_minutes = 0
    total_early_minutes = 0

    for r in records:
        if r.check_in and r.check_out:
            in_dt = datetime.combine(r.date, r.check_in)
            out_dt = datetime.combine(r.date, r.check_out)
            total_seconds_worked += (out_dt - in_dt).seconds

        # count check-ins per day
        checkins_per_day[r.date] = checkins_per_day.get(r.date, 0) + 1

        # parse late_by and early_by (HH:MM)
        if r.late_by and r.late_by != "00:00":
            h, m = map(int, r.late_by.split(":"))
            total_late_minutes += h*60 + m
        if r.early_by and r.early_by != "00:00":
            h, m = map(int, r.early_by.split(":"))
            total_early_minutes += h*60 + m

    avg_work_hours = (total_seconds_worked / 3600) / days_present if days_present > 0 else 0
    avg_checkins = sum(checkins_per_day.values()) / days_present if days_present > 0 else 0
    avg_late_minutes = total_late_minutes / days_present if days_present > 0 else 0
    avg_early_minutes = total_early_minutes / days_present if days_present > 0 else 0

    analysis = {
        "total_days": total_days,
        "days_present": days_present,
        "days_absent": days_absent,
        "attendance_percentage": round((days_present / total_days) * 100, 2),
        "avg_daily_work_hours": round(avg_work_hours, 2),
        "avg_checkins_per_day": round(avg_checkins, 2),
        "avg_late_minutes": round(avg_late_minutes, 2),
        "avg_early_leave_minutes": round(avg_early_minutes, 2)
    }

    return {"isSuccess": True, "employee": emp.name, "data": analysis}


# ==============================
if __name__ == "__main__":
    app.run(debug=True)
