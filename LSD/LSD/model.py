from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from LSD.database import Base


# ============================
# 👤 جدول المستخدمين
# ============================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(100), unique=True, nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    company_name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # علاقات
    scans = relationship("ScanLog", back_populates="user", cascade="all, delete")
    chats = relationship("ChatHistory", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User(user_name={self.user_name}, email={self.email})>"


# ============================
# 💬 جدول سجل المحادثات
# ============================
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # العلاقة العكسية
    user = relationship("User", back_populates="chats")

    def __repr__(self):
        return f"<ChatHistory(user_id={self.user_id}, question={self.question[:30]})>"


# ============================
# 🧠 جدول الأسئلة والأجوبة (قاعدة المعرفة)
# ============================
class QAPair(Base):
    __tablename__ = "questions_answers"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    def __repr__(self):
        return f"<QAPair(id={self.id})>"


# ============================
# 🧾 جدول سجلات الفحص
# ============================
class ScanLog(Base):
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    source_ip = Column(String)
    method = Column(String)
    url = Column(String)
    score = Column(Float)
    action = Column(String)
    reason = Column(String)
    matched_rules = Column(JSON)
    features = Column(JSON)

    # العلاقات
    user = relationship("User", back_populates="scans")
    results = relationship("Result", back_populates="scan", cascade="all, delete")

    def __repr__(self):
        return f"<ScanLog(id={self.id}, user_id={self.user_id}, score={self.score})>"


# ============================
# 📊 جدول نتائج الفحص
# ============================
class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("scan_logs.id", ondelete="CASCADE"))
    score = Column(Float, nullable=False)
    action = Column(String(100), nullable=False)
    reason = Column(String(255), nullable=True)
    matched_rules = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # العلاقة العكسية
    scan = relationship("ScanLog", back_populates="results")

    def __repr__(self):
        return f"<Result(scan_id={self.scan_id}, score={self.score})>"
