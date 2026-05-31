import pandas as pd
from sqlalchemy import create_engine, text
from core.config import Config

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)


def clean_text_for_mysql(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressive cleaning for MySQL encoding issues"""
    df = df.copy()

    text_cols = ['c_nom_projet', 'c_phase_en_cours', 'c_odd', 'c_normes_techniques_applicables',
                 'c_etude_impact_environnement', 'c_localite', 'c_projet_parent', 'c_owner']

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Replace problematic characters
            df[col] = df[col].str.replace(r'[\u2018\u2019\u201c\u201d\u2013\u2014]', "'", regex=True)  # quotes & dashes
            df[col] = df[col].str.replace('≤', '<=')
            df[col] = df[col].str.replace('≥', '>=')
            df[col] = df[col].str.replace('’', "'")
            df[col] = df[col].str.replace('–', '-')
            df[col] = df[col].str.replace('—', '-')
            # Remove any remaining non-ASCII if needed
            # df[col] = df[col].str.encode('ascii', errors='replace').str.decode('ascii')

    return df


def load_to_db(df: pd.DataFrame, table_name: str, if_exists='replace'):
    df_clean = clean_text_for_mysql(df)

    # Force UTF-8mb4 connection
    with engine.connect() as conn:
        conn.execute(text("SET NAMES utf8mb4"))
        conn.execute(text("SET CHARACTER SET utf8mb4"))

    df_clean.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=300,  # smaller chunks
        method=None  # disable multi for safety
    )

    print(f"✅ Successfully loaded {len(df_clean)} rows into → {table_name}")


# Optional: Drop table before loading to avoid column conflicts
def drop_and_load(df: pd.DataFrame, table_name: str):
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    load_to_db(df, table_name, if_exists='replace')