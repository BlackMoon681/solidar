from etl.transform import run_transformation
from etl.load import drop_and_load
from core.config import Config

if __name__ == "__main__":
    ppp_clean, pgd_clean, etude_clean = run_transformation()

    drop_and_load(ppp_clean, Config.TABLE_PPP_CLEAN)
    drop_and_load(pgd_clean, Config.TABLE_PGD_CLEAN)
    drop_and_load(etude_clean, Config.TABLE_PGD_ETUDE_CLEAN)

    print("\nAll clean tables created successfully!")