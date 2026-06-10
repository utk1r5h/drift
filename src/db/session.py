from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://utkarshshukla@localhost:5432/drift_observability"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()




def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
