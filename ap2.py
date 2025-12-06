# --------IMPORTS & CONFIGURATION------------------------------------------------------------------------------------------------------------------------------
from flask import Flask, jsonify, request, render_template, redirect, url_for, Response, session, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, time, timedelta
import time as t
from sqlalchemy import inspect, text
import os
import cv2
import numpy as np
import pyttsx3
from scipy.spatial import distance as dist
import dlib
from ultralytics import YOLO

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
spoof_model = YOLO(os.path.join(app.root_path, "spoof.pt"))
app.secret_key = "YOUR_SECRET_KEY"   # IMPORTANT


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///employees.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy()

db.init_app(app)
#-------------------------------------------------------------------------------------------------------------------------------

camera = None
blink_triggered = False
reset_camera = False

LEFT_EYE  = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark positions

BASE_PATH = os.path.join(app.root_path, "faceinfo")

predictor_path = "shape_predictor_68_face_landmarks.dat"  # download required
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


##############--------------------------------------------------------------------------------------------------------------------------------




# Haar Cascade face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



#################################################################################################

def require_login():
    if not session.get("logged_in"):
        return redirect("/login")
    return None


def require_role(role):
    if not session.get("logged_in"):
        return render_template("message.html",
                               message="Please login to continue.",
                               buttons=[{"text": "Login", "link": "/login"}])

    if session.get("user_role") != role:
        return render_template("message.html",
                               message="Access Denied! Admin only.",
                               buttons=[{"text": "Home", "link": "/"}])

    return None


#################################################################################################

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def recognize_employee(frame):
    # Load LBPH recognizer and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    label_dict = np.load("face_labels.npy", allow_pickle=True).item()

    # Haar face detector
    face_det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return ("no_face", None)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

        emp_no, confidence = recognizer.predict(roi)

        # Lower confidence = better match
        if confidence < 70:
            emp_id = label_dict[int(emp_no)]
            return ("recognized", emp_id)
        else:
            return ("unknown", None)

    return ("no_face", None)

def blink_and_spoof_detect(
    spoof_model_path="spoof.pt",
    ear_threshold=0.23,
    blink_frames=3,
    required_blinks=2
    ):

    # Load YOLO spoof model
    model = YOLO(spoof_model_path)

    # Dlib models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    
    # Load label dictionary
    label_dict = np.load("face_labels.npy", allow_pickle=True).item()


    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))

    cap = cv2.VideoCapture(0)

    blink_counter = 0
    blink_total = 0

    print("\n➡ Please blink twice...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for rect in faces:
            shape = predictor(gray, rect)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])

            # EAR for both eyes
            ear_left = eye_aspect_ratio(shape_np[LEFT_EYE])
            ear_right = eye_aspect_ratio(shape_np[RIGHT_EYE])
            ear = (ear_left + ear_right) / 2.0

            # Blink detection
            if ear < ear_threshold:
                blink_counter += 1
            else:
                if blink_counter >= blink_frames:
                    blink_total += 1
                    print(f"👁 Blink detected #{blink_total}")

                    if blink_total >= required_blinks:
                        print("📸 Capturing frame...")
                        captured = frame.copy()

                        cap.release()
                        cv2.destroyAllWindows()

                        # ------------------------------
                        # SPOOF DETECTION
                        # ------------------------------
                        print("\n🔍 Checking spoof...")

                        results = model(captured, verbose=False)
                        result = results[0]

                        cls = int(result.boxes.cls[0])
                        conf = float(result.boxes.conf[0])

                        if cls ==0 :
                            print(f"\n❌ FAKE FACE DETECTED (confidence: {conf:.2f})")
                            return "spoof"
                           
                        
                        if cls == 1:  # real face
                            print(f"\n✔ REAL FACE DETECTED (confidence: {conf:.2f})")
                        
                            # ----------- FACE RECOGNITION PART --------------
                            status, emp_id = recognize_employee(captured)
                        
                            if status == "recognized":
                                print(f"✅ Employee Recognized: {emp_id}")
                                return ("Recognized", emp_id)
                        
                            elif status == "unknown":
                                print("❌ Face not recognized.")
                                return ("unknown", None)
                        
                            else:
                                print("❌ No face detected in captured image.")
                                return ("no_face", None)


                blink_counter = 0

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    return "exit"

    


##########################################################################################


def process_checkinout(emp_id):
    emp = Employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        return "Employee not found"

    emp_name = emp.name
    today = datetime.now().date()

    record = Attendance.query.filter_by(
        emp_id=emp_id,
        date=today,
        check_out=None
    ).order_by(Attendance.id.desc()).first()

    # ✅ CHECK-IN
    if not record:
        record_checkin(emp_id, emp_name)
        return f"{emp_name} checked in successfully"

    # ✅ CHECK-OUT
    record_checkout(record)
    return f"{emp_name} checked out successfully"




# ==============================
# Employee Table
# ==============================
class Employee(db.Model):
    __tablename__ = 'employee'

    id = db.Column(db.Integer, primary_key=True)
    emp_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email_id = db.Column(db.String(100),unique=True , nullable=False)
    Passward = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    

    def __repr__(self):
        return f"<Employee {self.emp_id} - {self.name}>"


# ==============================
# Attendance Table
# ==============================
class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    emp_id = db.Column(db.String(50), db.ForeignKey('employee.emp_id'), nullable=False)
    emp_name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, default=lambda: datetime.now().date(), nullable=False)
    check_in = db.Column(db.Time)
    check_out = db.Column(db.Time)
    late_by = db.Column(db.String(10), default="00:00")   # HH:MM
    early_by = db.Column(db.String(10), default="00:00")

    def __repr__(self):
        return f"<Attendance {self.emp_id} on {self.date}>"


# ==============================
# Check-in / Check-out Functions
# ==============================


def record_checkin(emp_id, emp_name, official_start=time(9, 0), grace_minutes=30):
    today = datetime.now().date()
    now_time = datetime.now().time().replace(microsecond=0)

    now_dt = datetime.combine(today, now_time)
    start_dt = datetime.combine(today, official_start)
    grace_dt = start_dt + timedelta(minutes=grace_minutes)

    if now_dt <= grace_dt:
        late_by = "00:00"
    else:
        diff = now_dt - grace_dt
        hours, remainder = divmod(diff.seconds, 3600)
        minutes = remainder // 60
        late_by = f"{hours:02}:{minutes:02}"

    record = Attendance(
        emp_id=emp_id,
        emp_name=emp_name,
        date=today,
        check_in=now_time,
        late_by=late_by
    )

    db.session.add(record)
    db.session.commit()


def record_checkout(record, official_end=time(17, 0)):
    today = datetime.now().date()
    now_time = datetime.now().time().replace(microsecond=0)

    now_dt = datetime.combine(today, now_time)
    end_dt = datetime.combine(today, official_end)

    if now_dt >= end_dt:
        early_by = "00:00"
    else:
        diff = end_dt - now_dt
        hours, remainder = divmod(diff.seconds, 3600)
        minutes = remainder // 60
        early_by = f"{hours:02}:{minutes:02}"

    record.check_out = now_time
    record.early_by = early_by
    db.session.commit()



#------FACE RECOGNITION UTILITIES-----------------------------------------------------------------------------------------------------------------

def train_faces():
    base_dir = BASE_PATH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []
    label_dict = {}

    # --- Loop through folders (each employee) ---
    for person_folder in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_folder)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):

            # accept only images
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            parts = img_name.split(".")
            # Expected format: user.<emp_id>.<count>.jpg
            if len(parts) < 4:
                print(f"[WARN] Skipping invalid filename: {img_name}")
                continue

            emp_id = parts[1]  # String employee ID

            # Validate emp_id numeric
            try:
                numeric_label = int(emp_id)
            except ValueError:
                print(f"[WARN] Invalid emp_id in filename: {img_name}")
                continue

            img_path = os.path.join(person_path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue

            # Detect faces
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                continue

            # Save face ROIs
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))  # consistent size

                face_samples.append(face_resized)
                ids.append(numeric_label)
                label_dict[numeric_label] = emp_id

    # --- No faces found ---
    if not face_samples:
        print("[WARN] No face samples found. Training cancelled.")
        return

    # --- Train model from scratch ---
    recognizer.train(face_samples, np.array(ids))

    # --- Save label mapping and model ---
    np.save("face_labels.npy", label_dict)
    recognizer.save("trainer.yml")

    print("[INFO] Face model retrained successfully.")
    print(f"[INFO] Total faces trained: {len(face_samples)}")
    print(f"[INFO] Employees trained: {len(label_dict)}")


def capture_faces_for_employee(emp_name, emp_id):
    emp_name = emp_name.strip().replace(" ", "_")
    emp_id = str(emp_id)

    save_path = os.path.join(BASE_PATH, f"{emp_name}_{emp_id}")
    os.makedirs(save_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h + 10), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/50", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Capture Employee Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            if len(faces) == 0:
                continue

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                count += 1
                img_name = f"user.{emp_id}.{count}.jpg"
                cv2.imwrite(os.path.join(save_path, img_name), face_img)
                t.sleep(0.3)

        elif key == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {count} face samples saved for {emp_name}.")

#--------------------------------------------------------------------------------------------#------------------------------------------------------------------
#--------------------------------------------------------------------------------------------#------------------------------------------------------------------
#--------------------------------------------------------------------------------------------#-----------------------------------------------------------------
def recognize_employee_from_image(frame, threshold=70):
    if not os.path.exists("trainer.yml") or not os.path.exists("face_labels.npy"):
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = np.load("face_labels.npy", allow_pickle=True).item()

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)

        if confidence < threshold:
            return {
                "emp_id": label,
                "name": labels.get(label, "Unknown"),
                "confidence": confidence
            }

    return None

@app.route("/api/check_face", methods=["POST"])
def api_check_face():

    file = request.files.get("image")
    if not file:
        return {"isSuccess": False, "message": "Image required"}, 400

    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"isSuccess": False, "message": "Invalid image"}, 400

    status, emp_id = detect_and_recognize(frame)

    if status == "spoof":
        return {"isSuccess": False, "message": "Spoof attack detected"}, 403

    elif status in ["unknown", "no_face"]:
        return {
            "isSuccess": True,
            "message": "Face not recognized, you can add new employee"
        }, 201

    return {
        "isSuccess": True,
        "message": "Face recognized, employee already exists",
        "data": {"emp_id": emp_id}
    }, 200





@app.route("/admin/add_employee", methods=["POST"])
def add_employee():

    admin_id = request.form.get("admin_id")
    admin_pass = request.form.get("admin_pass")

    # Admin validation
    admin = Employee.query.filter_by(emp_id=admin_id, Passward=admin_pass, role="admin").first()
    if not admin:
        return {
            "isSuccess": False,
            "message": "Invalid admin credentials"
        }, 403

    emp_id = request.form.get("emp_id")
    name = request.form.get("name")
    email = request.form.get("email_id")
    password = request.form.get("password")
    role = request.form.get("role")

    # ✅ Validation
    if not all([emp_id, name, email, password, role]):
        return {
            "isSuccess": False,
            "message": "All fields are required"
        }, 400

    # ✅ Check duplicate
    existing = Employee.query.filter_by(emp_id=emp_id).first()
    if existing:
        return {
            "isSuccess": False,
            "message": f"Employee {emp_id} already exists"
        }, 409

    # ✅ Hash password

    # ✅ Create employee
    new_emp = Employee(
        emp_id=emp_id,
        name=name,
        email_id=email,
        Passward=password,
        role=role,
    )

    db.session.add(new_emp)
    db.session.commit()

    return {
        "isSuccess": True,
        "message": "Employee created successfully",
        "employee": {
            "emp_id": emp_id,
            "name": name,
            "email_id": email,
            "role": role
        }
    }, 201



@app.route("/api/train_faces", methods=["POST"])
def api_train_faces():
    emp_id = request.form.get("emp_id")
    emp_name = request.form.get("emp_name")

    if not emp_id or not emp_name:
        return {"isSuccess": False, "message": "emp_id and emp_name required"}, 400

    files = request.files.getlist("images")
    if not files:
        return {"isSuccess": False, "message": "No images uploaded"}, 400

    save_path = os.path.join(BASE_PATH, f"{emp_name}_{emp_id}")
    os.makedirs(save_path, exist_ok=True)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    for file in files:
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces = detector.detectMultiScale(img, 1.2, 5)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            count += 1
            cv2.imwrite(
                os.path.join(save_path, f"user.{emp_id}.{count}.jpg"),
                face
            )

    if count == 0:
        return {"isSuccess": False, "message": "No face detected"}, 400

    train_faces()

    return {
        "isSuccess": True,
        "message": f"{count} face samples saved & model trained"
    }, 201






# Paths
trainer_path = "trainer.yml"
labels_path = "face_labels.npy"

# 1️⃣ Create trainer.yml if missing
if not os.path.exists(trainer_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.save(trainer_path)
    print("[INFO] Created empty trainer.yml")
else:
    print("[INFO] trainer.yml already exists")

# 2️⃣ Create face_labels.npy if missing
if not os.path.exists(labels_path):
    np.save(labels_path, {})
    print("[INFO] Created empty face_labels.npy")
else:
    print("[INFO] face_labels.npy already exists")
    
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainer.yml")
label_dict = np.load("face_labels.npy", allow_pickle=True).item()

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# YOLO spoof detector (do this once at top of file)
spoof_model = YOLO("spoof.pt")


def rec_emp(frame):
    """Recognize employee from frame. Returns emp_id or None"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200))

    label_id, confidence = face_recognizer.predict(roi)
    if confidence > 70:  # adjust threshold if needed
        return None

    emp_id = label_dict.get(int(label_id))
    return emp_id


def is_spoof(frame):
    """Return True if spoof detected"""
    results = spoof_model(frame, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = spoof_model.names[cls_id].lower()
            if cls_name == "spoof":
                return True
    return False


def detect_and_recognize(frame):
    if is_spoof(frame):
        return ("spoof", None)

    emp_id = rec_emp(frame)
    if emp_id is not None:
        return ("recognized", emp_id)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return ("no_face", None)

    return ("unknown", None)



@app.route("/api/mark_attendance", methods=["POST"])
def mark_attendance():
    file = request.files.get("image")
    if not file:
        return {"isSuccess": False, "message": "Image required"}, 400

    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"isSuccess": False, "message": "Invalid image"}, 400

    status, emp_id = detect_and_recognize(frame)

    if status == "spoof":
        return {"isSuccess": False, "message": "Spoof attack detected"}, 403
    elif status in ["unknown", "no_face"]:
        return {"isSuccess": False, "message": "Face not recognized"}, 404

    message = process_checkinout(emp_id)
    return {"isSuccess": True, "message": message, "data": {"emp_id": emp_id}}, 200



@app.route("/api/employees", methods=["POST"])
def api_get_all_employees():
    
    admin_id = request.form.get("admin_id")
    admin_pass = request.form.get("admin_pass")

    # Admin validation
    admin = Employee.query.filter_by(emp_id=admin_id, Passward=admin_pass, role="admin").first()
    if not admin:
        return {
            "isSuccess": False,
            "message": "Invalid admin credentials"
        }, 403
    employees = Employee.query.all()

    data = [{
        "emp_id": emp.emp_id,
        "name": emp.name,
        "email_id": emp.email_id,
        "role": emp.role
    } for emp in employees]

    return {
        "isSuccess": True,
        "message": "Employees fetched successfully",
        "count": len(data),
        "employees": data
    }, 200


@app.route("/api/update_employee", methods=["POST"])
def api_update_employee():
    # Step 1: Admin login validation
    admin_id = request.form.get("admin_id")
    admin_pass = request.form.get("admin_pass")

    # Admin validation
    admin = Employee.query.filter_by(emp_id=admin_id, Passward=admin_pass, role="admin").first()
    if not admin:
        return {
            "isSuccess": False,
            "message": "Invalid admin credentials"
        }, 403

    # Step 2: Employee update
    emp_id = request.form.get("emp_id")
    employee = Employee.query.filter_by(emp_id=emp_id).first()

    if not employee:
        return {"isSuccess": False, "message": "Employee not found"}, 404

    # Fields to update
    employee.name = request.form.get("name", employee.name)
    employee.email_id = request.form.get("email_id", employee.email_id)
    employee.Passward = request.form.get("Passward", employee.Passward)
    employee.role = request.form.get("role", employee.role)

    db.session.commit()

    return {
        "isSuccess": True,
        "message": "Employee updated successfully",
        "employee": {
            "emp_id": employee.emp_id,
            "name": employee.name,
            "email_id": employee.email_id,
            "role": employee.role
        }
    }
@app.route("/api/attendance", methods=["POST"])
def api_attendance():
    # 1. Validate admin/user login
    admin_id = request.form.get("admin_id")
    admin_pass = request.form.get("admin_pass")

    admin = Employee.query.filter_by(emp_id=admin_id, Passward=admin_pass).first()
    if not admin:
        return {"isSuccess": False, "message": "Invalid credentials"}, 403

    role = admin.role
    logged_emp_id = admin.emp_id

    # 2. Date filter
    date_filter = request.form.get("date")
    if not date_filter:
        date_filter = datetime.now().strftime("%Y-%m-%d")

    target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()

    # 3. Employees list based on role
    if role == "admin":
        employees = Employee.query.all()
    else:
        employees = Employee.query.filter_by(emp_id=logged_emp_id).all()

    # 4. Fetch attendance for the same date
    records = Attendance.query.filter_by(date=target_date).all()

    # 5. Build JSON response
    attendance_data = []
    for emp in employees:
        rec = next((r for r in records if r.emp_id == emp.emp_id), None)
        attendance_data.append({
            "emp_id": emp.emp_id,
            "emp_name": emp.name,
            "check_in": rec.check_in.strftime("%H:%M") if rec and rec.check_in else None,
            "check_out": rec.check_out.strftime("%H:%M") if rec and rec.check_out else None,
            "late_by": rec.late_by if rec else None,
            "early_by": rec.early_by if rec else None,
            "status": (
                "Absent" if not rec else
                "Checked-in only" if rec.check_in and not rec.check_out else
                "Present"
            )
        })

    return {
        "isSuccess": True,
        "message": "Attadance success",
        "date": date_filter,
        "attendance": attendance_data
    }, 200
    

@app.route("/api/employee_analysis", methods=["POST"])
def api_employee_analysis():

    # ---------- ADMIN / USER LOGIN VALIDATION ----------
    admin_id = request.form.get("admin_id")
    admin_pass = request.form.get("admin_pass")

    admin = Employee.query.filter_by(emp_id=admin_id, Passward=admin_pass).first()
    if not admin:
        return {"isSuccess": "false", "message": "Invalid login credentials"}, 403

    role = admin.role
    logged_emp_id = admin.emp_id

    # -------------- FILTER INPUTS -----------------
    search = request.form.get("search_id", "").strip().lower()
    from_date = request.form.get("from_date")
    to_date = request.form.get("to_date")

    query = Attendance.query

    if role == "admin":
        if search:
            query = query.filter(
                (Attendance.emp_name.ilike(f"%{search}%")) |
                (Attendance.emp_id.ilike(f"%{search}%"))
            )
    else:
        # Normal employee: only view his own records
        query = query.filter_by(emp_id=logged_emp_id)

    if from_date:
        query = query.filter(Attendance.date >= from_date)
    if to_date:
        query = query.filter(Attendance.date <= to_date)

    records = query.order_by(Attendance.date).all()

    # ---------- AGGREGATION ----------
    total_late = sum(
        int(r.late_by.split(":")[0]) * 60 + int(r.late_by.split(":")[1])
        for r in records if r.late_by
    )

    total_early = sum(
        int(r.early_by.split(":")[0]) * 60 + int(r.early_by.split(":")[1])
        for r in records if r.early_by
    )

    def daily_hours(r):
        if not (r.check_in and r.check_out):
            return 0
        diff = datetime.combine(r.date, r.check_out) - datetime.combine(r.date, r.check_in)
        return diff.seconds / 3600

    total_hours = sum(daily_hours(r) for r in records)
    daily_avg = (total_hours / len(records)) if records else 0

    # ----------- FORMAT RESULTS -----------
    record_list = []
    for r in records:
        record_list.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "emp_id": r.emp_id,
            "emp_name": r.emp_name,
            "check_in": r.check_in.strftime("%H:%M") if r.check_in else None,
            "check_out": r.check_out.strftime("%H:%M") if r.check_out else None,
            "late_by": r.late_by,
            "early_by": r.early_by,
            "daily_hours": round(daily_hours(r), 2)
        })

    return {
        "isSuccess": "true",
        "role": role,
        "Admin_id": logged_emp_id,
        "filters_used": {
            "search": search,
            "from_date": from_date,
            "to_date": to_date
        },
        "summary": {
            "total_late_minutes": total_late,
            "total_early_minutes": total_early,
            "total_hours": round(total_hours, 2),
            "daily_average_hours": round(daily_avg, 2)
        },
        "records": record_list
    }, 200
    

@app.route('/api/delete_employee/<int:emp_id>', methods=['POST'])
def delete_employee_by_emp_id(emp_id):
   

    employee = Employee.query.filter_by(emp_id=emp_id).first()

    print("👉 employee found:", employee)

    if not employee:
        return {
            "isSuccess": False,
            "message": "Employee not found in DB"
        }, 404

    db.session.delete(employee)
    db.session.commit()

    return {
        "isSuccess": True,
        "message": f"Employee {emp_id} deleted"
    }, 200

#--------------------------------------------------------------------------------------------#------------------------------------------------------------------
# ----- ROUTES---------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template("checkinout.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        emp_id = request.form.get("emp_id")
        password = request.form.get("Passward")

        emp = Employee.query.filter_by(emp_id=emp_id, Passward=password).first()
        if not emp:
            return render_template("message.html",
                                   message="Invalid ID or Password.",
                                   buttons=[{"text": "Try Again", "link": "/login"}])

        session["logged_in"] = True
        session["emp_id"] = emp.emp_id
        session["user_role"] = emp.role

        return redirect("/")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")



@app.route("/checkinout")
def checkinout():
    if not os.path.exists("trainer.yml") or not os.path.exists("face_labels.npy"):
        return render_template("message.html",
            message="Face model not trained yet. Please add an employee first.",
            buttons=[{"text": "Add Employee", "link": "/new_employee"}]
        )    
    return render_template("checkinout.html")




@app.route('/video_feed')
def video_feed():

    def generate():
        with app.app_context():

            result = blink_and_spoof_detect()

            # ✅ EMPLOYEE RECOGNIZED
            if isinstance(result, tuple) and result[0] == "Recognized":
                status, emp_id = result

                emp = Employee.query.filter_by(emp_id=str(emp_id)).first()
                if not emp:
                    msg = "Employee not found!"
                else:
                    emp_name = emp.name
                    today = datetime.now().date()

                    # ✅ FIND LATEST OPEN SESSION
                    open_record = Attendance.query.filter_by(
                        emp_id=emp_id,
                        date=today,
                        check_out=None
                    ).order_by(Attendance.id.desc()).first()

                    # ✅ CHECK-IN
                    if not open_record:
                        record_checkin(emp_id, emp_name)
                        msg = f"{emp_name} checked in successfully!"
                        pyttsx3.speak(msg)

                    # ✅ CHECK-OUT
                    else:
                        record_checkout(open_record)
                        msg = f"{emp_name} checked out successfully!"
                        pyttsx3.speak(msg)

                frame = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(
                    frame, msg, (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2
                )

                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')
                return

            # ❌ SPOOF ATTACK
            if result == "spoof":
                frame = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(frame, "❌ Spoof attack detected!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')
                return

            # ❌ UNKNOWN FACE
            if isinstance(result, tuple) and result[0] == "unknown":
                frame = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(frame, "❌ Face not recognized.", (70, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')
                return

            # ❌ NO FACE
            if isinstance(result, tuple) and result[0] == "no_face":
                frame = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(frame, "❌ No face detected.", (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')
                return

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def recognize_employee_from_image(frame, threshold=70):
    import cv2
    import numpy as np
    import os

    if not os.path.exists("trainer.yml") or not os.path.exists("face_labels.npy"):
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = np.load("face_labels.npy", allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        # ✅ LOWER confidence = better match
        if confidence < threshold:
            return {
                "emp_id": label,
                "name": labels[label],
                "confidence": confidence
            }

    return None

@app.route("/check_face", methods=["GET"])
def check_face():
    cap = cv2.VideoCapture(0)

    matched_employee = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = recognize_employee_from_image(frame)

        cv2.imshow("Face Verification - Press Q", frame)

        if result:
            matched_employee = result
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ✅ FACE MATCH FOUND
    if matched_employee:
        return render_template(
            "message.html",
            message=f"❌ Employee already exists \n Employee ID:{matched_employee['emp_id']} \n Confidence:{matched_employee['confidence']:.2f}",
            buttons=[
                {"text": "Employees ", "link": "employees"}
            ]
        )

    # ❌ NO FACE FOUND → GO TO NEW EMPLOYEE FORM
    return redirect(url_for("new_employee"))


@app.route("/new_employee", methods=["GET", "POST"])
def new_employee():
    check = require_role("admin")
    if check:
        return check

    if request.method == "POST":
        name = request.form.get("name")
        emp_id = request.form.get("emp_id")
        email_id = request.form.get("email_id")
        password = request.form.get("Passward")
        role = request.form.get("role")

        new_emp = Employee(
            name=name,
            emp_id=emp_id,
            email_id=email_id,
            Passward=password,
            role=role
        )

        db.session.add(new_emp)
        db.session.commit()

        return render_template(
            "message.html",
            message="✅ New employee saved. Now capture face.",
            buttons=[
                {"text": "Capture Face", "link": f"/capture_face/{emp_id}"},
                {"text": "Home", "link": "/"}
            ]
        )

    return render_template("new_employee.html")





@app.route("/capture_face/<emp_id>")
def capture_face(emp_id):
    employee = Employee.query.filter_by(emp_id=emp_id).first()
    if not employee:
        return render_template("message.html", message="Employee not found.")

    capture_faces_for_employee(employee.name, employee.emp_id)
    train_faces()

    return render_template(
        "message.html",
        message=f"Face data captured and model trained for {employee.name}!",
        buttons=[
            {"text": "Add Another", "link": "/new_employee"},
            {"text": "Home", "link": "/"}
        ]
    )
    
@app.route("/update_employee/<emp_id>", methods=["GET", "POST"])
def update_employee(emp_id):
    check = require_role("admin")
    if check:
        return check

    employee = Employee.query.filter_by(emp_id=emp_id).first()
    if not employee:
        return render_template("message.html", message="Employee not found.")

    if request.method == "POST":
        employee.name = request.form.get("name")
        employee.email_id = request.form.get("email_id")
        employee.Passward = request.form.get("Passward")
        employee.role = request.form.get("role")
        db.session.commit()

        # Send updated details to device (SOAP)
        soap_result = send_employee_to_device(
            username="admin",
            password="admin123",
            emp_id=emp_id,
            name=employee.name,
            location="Pune",
            role=employee.role,
            verify_type="Face"
        )

        device_status = "Device updated successfully!" if "UpdateEmployeeResult" in soap_result else "Failed to update device."

        return render_template("message.html",
                               message=f"Employee updated! {device_status}",
                               detail=soap_result,
                               buttons=[{"text": "Home", "link": "/"}])

    return render_template("update_employee.html", employee=employee)






@app.route("/attendance")
def attendance():
    check = require_login()
    if check:
        return check

    role = session.get("user_role")
    logged_emp_id = str(session.get("emp_id"))  # ✅ force string

    date_filter = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()

    if role == "admin":
        employees = Employee.query.all()
    else:
        employees = Employee.query.filter_by(emp_id=logged_emp_id).all()

    records = Attendance.query.filter_by(date=target_date).all()

    attendance_data = []

    for emp in employees:
        emp_id_str = str(emp.emp_id)  # ✅ force string

        emp_records = [
            r for r in records
            if str(r.emp_id) == emp_id_str
        ]

        if emp_records:
            first_check_in = min(
                (r.check_in for r in emp_records if r.check_in),
                default=None
            )

            last_check_out = max(
                (r.check_out for r in emp_records if r.check_out),
                default=None
            )

            late_by = emp_records[0].late_by
            early_by = emp_records[-1].early_by
        else:
            first_check_in = None
            last_check_out = None
            late_by = "—"
            early_by = "—"

        attendance_data.append({
            "emp_id": emp.emp_id,
            "emp_name": emp.name,
            "check_in": first_check_in.strftime("%H:%M") if first_check_in else "—",
            "check_out": last_check_out.strftime("%H:%M") if last_check_out else "—",
            "late_by": late_by,
            "early_by": early_by
        })

    return render_template(
        "attendance.html",
        attendance_data=attendance_data,
        date_filter=date_filter
    )





@app.route('/delete/<int:id>', methods=['POST'])
def delete_employee(id):
    employee = Employee.query.get(id)

    if not employee:
        return render_template(      
            "message.html",
            message=f"Employee not found!",
            buttons=[
                {"text": "Return", "link": "/employees"}
            ]
        )

    db.session.delete(employee)
    db.session.commit()

    flash("Employee deleted successfully", "success")
    return render_template(      
            "message.html",
            message=f"Employee removed ",
            buttons=[
                {"text": "Return", "link": "/employees"}
            ]
        )


@app.route("/employee_analysis", methods=["GET"])
def employee_analysis():
    check = require_login()
    if check: return check

    role = session.get("user_role")
    logged_emp_id = session.get("emp_id")

    search = request.args.get("q", "").strip().lower()
    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")

    query = Attendance.query

    if role == "admin":
        if search:
            query = query.filter(
                (Attendance.emp_name.ilike(f"%{search}%")) |
                (Attendance.emp_id.ilike(f"%{search}%"))
            )
    else:
        query = query.filter_by(emp_id=logged_emp_id)

    if from_date:
        query = query.filter(Attendance.date >= from_date)
    if to_date:
        query = query.filter(Attendance.date <= to_date)

    records = query.order_by(Attendance.date).all()

    # -------------------------------
    #  AGGREGATION LOGIC (same code)
    # -------------------------------
    total_late = sum(
        int(r.late_by.split(":")[0]) * 60 + int(r.late_by.split(":")[1])
        for r in records if r.late_by
    )

    total_early = sum(
        int(r.early_by.split(":")[0]) * 60 + int(r.early_by.split(":")[1])
        for r in records if r.early_by
    )

    def daily_hours(r):
        if not (r.check_in and r.check_out):
            return 0
        diff = datetime.combine(r.date, r.check_out) - datetime.combine(r.date, r.check_in)
        return diff.seconds / 3600

    total_hours = sum(daily_hours(r) for r in records)
    daily_avg = (total_hours / len(records)) if records else 0

    return render_template(
        "employee_analysis.html",
        records=records,
        emp_id=records[0].emp_id if records else logged_emp_id,
        total_late=total_late,
        total_early=total_early,
        weekly_hours=round(total_hours, 2),
        monthly_hours=round(total_hours, 2),
        daily_avg=round(daily_avg, 2),
        search_query=search if role == "admin" else "",
        from_date=from_date,
        to_date=to_date,
        datetime=datetime
    )


@app.route("/employees")
def view_employees():
    
    employees = Employee.query.all()
    return render_template("essl_employee_list.html", employees=employees)


@app.route("/stop_camera")
def stop_camera():
    try:
        global cap
        cap.release()
    except:
        pass
    return "Camera stopped"

####################################################################################################



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
        
if __name__ == '__main__':
    app.run(debug=True)
