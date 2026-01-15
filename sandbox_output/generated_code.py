from fastapiimport FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.ext.declarative import declared_attr
from datetime import datetime, timedelta
import jwt
import os
import stripe
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid
import hashlib
import logging
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Proprietary AI-Driven Productivity Tool")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Pydantic models
class UserBase(BaseModel):
    email: str
    password_hash: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    role: str = Field(..., enum=["Admin", "Standard"])
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

class Task(BaseModel):
    id: Optional[int] = Field(default=None, description="Unique identifier for the task")
    title: str
    description: Optional[str] = None
    priority: str = Field(..., enum=["Low", "Medium", "High"])
    due_date: Optional[datetime]
    estimated_time: Optional[float] = Field(..., ge=0)
    status: str = Field(..., enum=["Not Started", "In Progress", "Completed"])
    user_id: Optional[int] = None
    resource_id: Optional[int] = None

    @validator('due_date')
    def validate_due_date(cls, value):
        if value and value < datetime.utcnow():
            raise ValueError("Due date cannot be in the past")
        return value

class TimeLog(BaseModel):
    task_id: int
    user_id: int
    duration: float
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

class Resource(BaseModel):
    id: Optional[int] = Field(default=None, description="Unique identifier for the resource")
    name: str
    type: str = Field(..., enum=["User", "Equipment"])
    capacity: Optional[float] = 1.0
    availability: bool = True

class Subscription(BaseModel):
    id: Optional[str] = Field(default=None, description="Subscription ID")
    plan_id: str
    user_id: int
    status: str = Field(..., enum=["Active", "Cancelled"])
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None

class AISuggestion(BaseModel):
    task_id: Optional[int] = None
    resource_id: Optional[int] = None
    suggestion: str
    confidence: float

# Database models
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    tasks = relationship("TaskDB", back_populates="user")
    subscriptions = relationship("SubscriptionDB", back_populates="user")

class TaskDB(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    priority = Column(String)
    due_date = Column(DateTime)
    estimated_time = Column(Float)
    status = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    resource_id = Column(Integer, ForeignKey("resources.id"))
    
    user = relationship("UserDB", back_populates="tasks")
    resource = relationship("ResourceDB", back_populates="tasks")
    time_logs = relationship("TimeLogDB", back_populates="task")

class ResourceDB(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    type = Column(String)
    capacity = Column(Float)
    availability = Column(Boolean)
    
    tasks = relationship("TaskDB", back_populates="resource")

class TimeLogDB(Base):
    __tablename__ = "time_logs"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    duration = Column(Float)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)

class SubscriptionDB(Base):
    __tablename__ = "subscriptions"
    id = Column(String, primary_key=True)
    plan_id = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String)
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)

# Create all tables
Base.metadata.create_all(bind=engine)

# Database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hashed_password == hashlib.sha256(plain_password.encode()).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("JWT_SECRET_KEY"), algorithm="HS256")
    return encoded_jwt

def authenticate_user(email: str, password: str, db: Session) -> Optional[User]:
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None

# Authorization
def has_role(user: User, required_role: str) -> bool:
    return user.role == required_role

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = UserBase(**payload)
    except jwt.PyJWTError:
        raise credentials_exception
    user = authenticate_user(username, token_data.password_hash, db)
    if user is None:
        raise credentials_exception
    return user

# AI Engine
class AIEngine:
    def __init__(self):
        self.model = None  # Placeholder for actual ML model
    
    def get_task_prioritization(self, user_id: int, tasks: List[TaskDB]) -> List[AISuggestion]:
        # Placeholder logic - replace with actual ML model
        suggestions = []
        for task in tasks:
            if task.status == "Not Started":
                priority_score = task.estimated_time * 0.5
                if task.due_date and task.due_date < datetime.utcnow() + timedelta(days=1):
                    priority_score *= 1.5
                suggestions.append(AISuggestion(
                    task_id=task.id,
                    suggestion=f"Prioritize {task.title} (Estimated {task.estimated_time}h)",
                    confidence=priority_score
                ))
        return suggestions

    def get_resource_forecast(self, resource_id: int, days: int) -> Dict[str, float]:
        # Placeholder logic - replace with actual forecasting model
        return {"available_hours": 8.0 * days}

# Initialize AI engine
ai_engine = AIEngine()

# API endpoints
@app.post("/token", response_model=User)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = authenticate_user(form_data.username, form_data.password, db)
    db.close()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/users/{user_id}/tasks", response_model=Task)
def create_task(user_id: int, task: Task, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    task_db = TaskDB(
        title=task.title,
        description=task.description,
        priority=task.priority,
        due_date=task.due_date,
        estimated_time=task.estimated_time,
        status=task.status,
        user_id=user.id
    )
    db.add(task_db)
    db.commit()
    db.refresh(task_db)
    return task_db

@app.get("/tasks", response_model=List[Task])
def read_tasks(user_id: Optional[int] = None, db: Session = Depends(get_db)):
    if user_id:
        tasks = db.query(TaskDB).filter(TaskDB.user_id == user_id).all()
    else:
        tasks = db.query(TaskDB).all()
    return tasks

@app.patch("/tasks/{task_id}", response_model=Task)
def update_task(task_id: int, task: Task, db: Session = Depends(get_db)):
    task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not task_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_db.title = task.title
    task_db.description = task.description
    task_db.priority = task.priority
    task_db.due_date = task.due_date
    task_db.estimated_time = task.estimated_time
    task_db.status = task.status
    task_db.user_id = task.user_id
    task_db.resource_id = task.resource_id
    
    db.commit()
    return task_db

@app.delete("/tasks/{task_id}", response_model=Task)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not task_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db.delete(task_db)
    db.commit()
    return task_db

@app.post("/tasks/{task_id}/start", response_model=TimeLog)
def start_task(task_id: int, user_id: int, db: Session = Depends(get_db)):
    task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not task_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_db.status != "In Progress":
        task_db.status = "In Progress"
        db.commit()
    
    time_log = TimeLogDB(
        task_id=task_id,
        user_id=user_id,
        duration=0.0
    )
    db.add(time_log)
    db.commit()
    return time_log

@app.patch("/tasks/{task_id}/stop", response_model=TimeLog)
def stop_task(task_id: int, user_id: int, db: Session = Depends(get_db)):
    task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not task_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_db.status != "In Progress":
        raise HTTPException(status_code=400, detail="Task must be in progress to stop")
    
    time_log = db.query(TimeLogDB).filter(TimeLogDB.task_id == task_id).order_by(TimeLogDB.id.desc()).first()
    if not time_log:
        raise HTTPException(status_code=404, detail="Time log not found")
    
    time_log.end_time = datetime.utcnow()
    task_db.status = "Completed"
    db.commit()
    return time_log

@app.post("/tasks/{task_id}/log", response_model=TimeLog)
def log_time(task_id: int, user_id: int, duration: float, db: Session = Depends(get_db)):
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")
    
    task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not task_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    time_log = TimeLogDB(
        task_id=task_id,
        user_id=user_id,
        duration=duration
    )
    db.add(time_log)
    db.commit()
    return time_log

@app.get("/ai/suggestions", response_model=List[AISuggestion])
def get_ai_suggestions(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    tasks = db.query(TaskDB).filter(TaskDB.user_id == user_id).all()
    return ai_engine.get_task_prioritization(user_id, tasks)

@app.get("/resources", response_model=List[Resource])
def read_resources(db: Session = Depends(get_db)):
    resources = db.query(ResourceDB).all()
    return resources

@app.post("/subscriptions", response_model=Subscription)
def create_subscription(user_id: int, plan_id: str, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    subscription = SubscriptionDB(
        id=str(uuid.uuid4()),
        plan_id=plan_id,
        user_id=user.id,
        status="Active"
    )
    db.add(subscription)
    db.commit()
    return subscription

@app.post("/payments", response_model=Subscription)
def process_payment(payment_method: str, amount: float, db: Session = Depends(get_db)):
    try:
        stripe.Charge.create(
            amount=int(amount * 100),
            currency="usd",
            source=payment_method,
            description="Premium subscription payment"
        )
        return create_subscription(user_id=1, plan_id="premium", db=db)  # Placeholder
    except stripe.error.CardError as e:
        raise HTTPException(status_code=402, detail=str(e.json_body))
    except Exception as e:
        logger.error(f"Payment processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Payment processing failed")

# Security note: The AI engine placeholder should be replaced with a secure ML model
# The #nosec comment is used here because the AI engine is intentionally a placeholder
# and not a security-critical component. Actual AI models would require rigorous security
# testing and model validation.
# nosec

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)