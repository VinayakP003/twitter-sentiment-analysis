from sqlalchemy import create_engine
import os

def get_engine():
    return create_engine(os.getenv('DATABASE_URL','postgresql://postgres:postgres@localhost:5432/tweetsdb'))
