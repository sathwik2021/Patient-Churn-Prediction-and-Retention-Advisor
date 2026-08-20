"""
Database Module — Patient Churn Prediction (SQLite)
===================================================
Manages users, predictions, and cohort datasets via SQLAlchemy & SQLite.
"""

import os
import hashlib
import secrets
import uuid
from datetime import timezone
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

DB_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = f"sqlite:///{os.path.join(DB_DIR, 'patient_churn_prediction.db')}"

# Use SQLite database
DATABASE_URL = os.getenv("DATABASE_URL", SQLITE_DB_PATH)

engine = create_engine(
    DATABASE_URL if DATABASE_URL.startswith("sqlite") else SQLITE_DB_PATH,
    connect_args={"check_same_thread": False} if "sqlite" in (DATABASE_URL or SQLITE_DB_PATH) else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"))
    patient_data = Column(Text)
    probability = Column(Float)
    risk_level = Column(String)
    primary_reason = Column(String)
    retention_advice = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class CohortDataset(Base):
    __tablename__ = "cohort_datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"))
    filename = Column(String)
    total_patients = Column(Integer)
    high_risk = Column(Integer)
    medium_risk = Column(Integer)
    low_risk = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class Session(Base):
    __tablename__ = "sessions"
    token = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def _pbkdf2(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 100_000
    ).hex()
    return f"{salt}${digest}"


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    return _pbkdf2(password, salt)


def verify_password(password: str, stored: str) -> bool:
    if "$" in stored:
        return _pbkdf2(password, stored.split("$", 1)[0]) == stored
    # Legacy unsalted SHA-256 hashes from older versions.
    return hashlib.sha256(password.encode()).hexdigest() == stored


def create_session(token: str, user_id: str):
    db = SessionLocal()
    try:
        db.add(Session(token=token, user_id=user_id))
        db.commit()
    finally:
        db.close()


def get_user_id_by_token(token: str):
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.token == token).first()
        return session.user_id if session else None
    finally:
        db.close()


def delete_session(token: str):
    db = SessionLocal()
    try:
        db.query(Session).filter(Session.token == token).delete()
        db.commit()
    finally:
        db.close()


def create_user(name: str, email: str, password: str) -> dict:
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return None
        new_user = User(id=user_id, name=name, email=email, password_hash=pw_hash)
        db.add(new_user)
        db.commit()
        return {"id": user_id, "name": name, "email": email}
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()


def authenticate_user(email: str, password: str) -> dict:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and verify_password(password, user.password_hash):
            return {"id": user.id, "name": user.name, "email": user.email}
        return None
    finally:
        db.close()


def get_user_by_id(user_id: str) -> dict:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return {"id": user.id, "name": user.name, "email": user.email}
        return None
    finally:
        db.close()


def save_prediction(user_id: str, patient_data: str, probability: float,
                    risk_level: str, primary_reason: str, retention_advice: str):
    db = SessionLocal()
    try:
        pred = Prediction(
            user_id=user_id,
            patient_data=patient_data,
            probability=probability,
            risk_level=risk_level,
            primary_reason=primary_reason,
            retention_advice=retention_advice
        )
        db.add(pred)
        db.commit()
    finally:
        db.close()


def save_cohort(user_id: str, filename: str, total: int, high: int, med: int, low: int):
    db = SessionLocal()
    try:
        cohort = CohortDataset(
            user_id=user_id,
            filename=filename,
            total_patients=total,
            high_risk=high,
            medium_risk=med,
            low_risk=low
        )
        db.add(cohort)
        db.commit()
    finally:
        db.close()


def get_user_predictions(user_id: str, limit: int = 50) -> list:
    db = SessionLocal()
    try:
        preds = db.query(Prediction).filter(Prediction.user_id == user_id)\
                    .order_by(Prediction.created_at.desc())\
                    .limit(limit).all()
        return [{
            "id": p.id,
            "user_id": p.user_id,
            "patient_data": p.patient_data,
            "probability": p.probability,
            "risk_level": p.risk_level,
            "primary_reason": p.primary_reason,
            "retention_advice": p.retention_advice,
            "created_at": p.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if p.created_at else None
        } for p in preds]
    finally:
        db.close()


def get_user_analytics(user_id: str) -> dict:
    db = SessionLocal()
    try:
        preds = db.query(Prediction).filter(Prediction.user_id == user_id).all()
        cohorts = db.query(CohortDataset).filter(CohortDataset.user_id == user_id)\
                .order_by(CohortDataset.created_at.desc()).all()

        probabilities = [p.probability for p in preds] if preds else []
        return {
            "total_evaluated": len(preds),
            "avg_churn": round(sum(probabilities) / len(probabilities) * 100, 1) if probabilities else 0,
            "high_risk_count": sum(1 for p in preds if p.risk_level == "High"),
            "medium_risk_count": sum(1 for p in preds if p.risk_level == "Medium"),
            "low_risk_count": sum(1 for p in preds if p.risk_level == "Low"),
            "cohort_uploads": [{
                "id": c.id,
                "filename": c.filename,
                "total_patients": c.total_patients,
                "high_risk": c.high_risk,
                "medium_risk": c.medium_risk,
                "low_risk": c.low_risk,
                "created_at": c.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if c.created_at else None
            } for c in cohorts],
        }
    finally:
        db.close()


# Auto-init tables
init_db()
