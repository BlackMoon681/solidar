"""
etl/load.py
-----------
Targets the cleaned_dw warehouse. Creates the database if needed and loads
dimension + fact tables with UTF-8mb4 safety.
"""
import pandas as pd
from sqlalchemy import create_engine, text
from core.config import Config

# Source (jwdb) and warehouse (cleaned_dw) engines
src_engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
dw_engine  = create_engine(Config.SQLALCHEMY_DW_URI)


def ensure_warehouse():
    """Create the cleaned_dw database if it does not exist."""
    server = create_engine(Config.SQLALCHEMY_SERVER_URI)
    with server.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME_DW} "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        conn.commit()
    server.dispose()
    print(f"Warehouse ready: {Config.DB_NAME_DW}")


def _clean_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = (df[col].astype(str)
                   .str.replace(r"[‘’]", "'", regex=True)
                   .str.replace(r"[“”]", '"', regex=True)
                   .str.replace(r"[–—]", "-", regex=True))
        df[col] = df[col].replace({"None": None, "nan": None})
    return df


def drop_and_load(df: pd.DataFrame, table_name: str, engine=None):
    engine = engine or dw_engine
    df_clean = _clean_text_cols(df)
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.commit()
    df_clean.to_sql(
        name=table_name, con=engine, if_exists="replace",
        index=False, chunksize=300, method=None,
    )
    print(f"  loaded {len(df_clean):>5} rows -> {Config.DB_NAME_DW}.{table_name}")


def load_dimensions(dim_frames: dict):
    print("Loading dimensions:")
    for name, frame in dim_frames.items():
        drop_and_load(frame, name)


def load_facts(ppp_clean, pgd_clean, etude_clean):
    print("Loading facts:")
    drop_and_load(ppp_clean,   Config.TABLE_PPP_CLEAN)
    drop_and_load(pgd_clean,   Config.TABLE_PGD_CLEAN)
    drop_and_load(etude_clean, Config.TABLE_PGD_ETUDE_CLEAN)
