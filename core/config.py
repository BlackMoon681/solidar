import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT"))
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")                       # source app DB (jwdb)
    DB_NAME_DW = os.getenv("DB_NAME_DW", "cleaned_dw")   # analytical warehouse

    # Source DB (Joget — raw tables live here)
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

    # Data warehouse (cleaned facts + dimensions)
    SQLALCHEMY_DW_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_DW}?charset=utf8mb4"

    # Server-level URI (no database selected — used to CREATE the warehouse)
    SQLALCHEMY_SERVER_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?charset=utf8mb4"

    # Raw source tables (jwdb)
    TABLE_PPP       = "app_fd_base_ppp"
    TABLE_PGD       = "app_fd_base_pgd"
    TABLE_PGD_ETUDE = "app_fd_base_pgd_etude"

    # Clean fact tables (cleaned_dw)
    TABLE_PPP_CLEAN       = "fact_ppp"
    TABLE_PGD_CLEAN       = "fact_pgd"
    TABLE_PGD_ETUDE_CLEAN = "fact_pgd_etude"
