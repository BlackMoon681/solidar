import pandas as pd
from sqlalchemy import create_engine
from core.config import Config

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

def extract_from_db(table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} rows from {table_name}")
    return df

