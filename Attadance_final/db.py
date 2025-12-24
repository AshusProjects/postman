# db.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, time, timedelta
from sqlalchemy import PickleType

db = SQLAlchemy()

# ==============================
# Employee Table
# ==============================
class Employee(db.Model):
    __tablename__ = 'employee'

    id = db.Column(db.Integer, primary_key=True)
    emp_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email_id = db.Column(db.String(100), unique=True, nullable=False)
    Passward = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    embedding = db.Column(db.PickleType, nullable=True)   # store numpy array of face embedding

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
# Attendance Functions
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

    # CHECK-IN
    if not record:
        record_checkin(emp_id, emp_name)
        return f"{emp_name} checked in successfully"

    # CHECK-OUT
    record_checkout(record)
    return f"{emp_name} checked out successfully"
