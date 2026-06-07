"""
etl/main.py
-----------
Builds the cleaned_dw warehouse end-to-end:

  1. Ensure the cleaned_dw database exists
  2. Build & load dimension tables (dim_*) from jwdb param/geo tables
  3. Clean the 3 facts, resolving PPP foreign keys to readable labels
  4. Load fact tables (fact_ppp, fact_pgd, fact_pgd_etude)

Run:  python -m etl.main
"""
from etl.dimensions import build_dimensions
from etl.transform import run_transformation
from etl.load import ensure_warehouse, load_dimensions, load_facts, src_engine
from core.config import Config


def main():
    print("=" * 60)
    print("  ETL → cleaned_dw  (galaxy schema: facts + dimensions)")
    print("=" * 60)

    # 1. Warehouse
    ensure_warehouse()

    # 2. Dimensions
    print("\nBuilding dimensions from jwdb:")
    dim_frames, label_maps = build_dimensions(src_engine)
    load_dimensions(dim_frames)

    # 3 + 4. Facts (with FK label resolution) -> warehouse
    print()
    ppp_clean, pgd_clean, etude_clean = run_transformation(label_maps)
    print()
    load_facts(ppp_clean, pgd_clean, etude_clean)

    print("\n" + "=" * 60)
    print(f"  Done. {len(dim_frames)} dimensions + 3 facts in {Config.DB_NAME_DW}")
    print("=" * 60)


if __name__ == "__main__":
    main()
