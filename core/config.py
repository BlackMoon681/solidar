import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT"))
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")

    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Raw source tables
    TABLE_PPP       = "app_fd_base_ppp"
    TABLE_PGD       = "app_fd_base_pgd"
    TABLE_PGD_ETUDE = "app_fd_base_pgd_etude"

    # Clean output tables
    TABLE_PPP_CLEAN       = "app_fd_ppp_clean"
    TABLE_PGD_CLEAN       = "app_fd_pgd_clean"
    TABLE_PGD_ETUDE_CLEAN = "app_fd_pgd_etude_clean"
