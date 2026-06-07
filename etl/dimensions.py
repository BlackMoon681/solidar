"""
etl/dimensions.py
-----------------
Single source of truth for the cleaned_dw dimension tables.

Each dimension is built from a Joget param/reference table in jwdb, cleaned
to (key, label[, parent_fk]) form, and loaded into cleaned_dw as `dim_*`.

The PPP fact references most dimensions by UUID; geographic dimensions are
referenced by integer code.  PGD / PGD-Etude store the label directly as text
(already denormalised) so their dims are emitted for completeness and joinable
on the label column.
"""
import pandas as pd
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Dimension specifications
#   target            : dim table name in cleaned_dw
#   source            : source table in jwdb
#   key               : source PK column           -> dim column "key"
#   label             : source label column          -> dim column "label"
#   label_ar          : optional arabic label        -> dim column "label_ar"
#   parent            : optional (src_col, dim_col)   parent FK kept in the dim
#   fact_fk           : column in the PPP fact that references this dim (or None)
# ---------------------------------------------------------------------------
DIM_SPECS = [
    # ---- PPP UUID dimensions ------------------------------------------------
    {"target": "dim_axe",                 "source": "app_fd_param_axe_ppp",
     "key": "id", "label": "c_intitule",                "fact_fk": "c_axe"},
    {"target": "dim_sous_axe",            "source": "app_fd_param_sous_axe_ppp",
     "key": "id", "label": "c_intitule_sous_axe",
     "parent": ("c_axe", "axe_key"),                    "fact_fk": "c_sous_axe"},
    {"target": "dim_classification",      "source": "app_fd_param_classification",
     "key": "id", "label": "c_classification",          "fact_fk": "c_classification"},
    {"target": "dim_sous_classification", "source": "app_fd_param_sous_class_ppp",
     "key": "id", "label": "c_sous_classification",
     "parent": ("c_classification", "classification_key"), "fact_fk": "c_sous_classification"},
    {"target": "dim_acteur_implementation","source": "app_fd_param_act_imp_ppp",
     "key": "id", "label": "c_acteur_implementation",   "fact_fk": "c_acteur_dimplementation"},
    {"target": "dim_delimitation",        "source": "app_fd_param_delim_geo_ppp",
     "key": "id", "label": "c_delimitation",            "fact_fk": "c_delimination"},
    {"target": "dim_initiateur",          "source": "app_fd_param_initiateur_ppp",
     "key": "id", "label": "c_initiateur",              "fact_fk": "c_initiateur"},
    {"target": "dim_owner",               "source": "app_fd_param_owner_ppp",
     "key": "id", "label": "c_owner",                   "fact_fk": "c_owner"},
    {"target": "dim_mode_financement",    "source": "app_fd_param_mode_finance",
     "key": "id", "label": "c_mode_financement",        "fact_fk": "c_mode_financement"},
    {"target": "dim_mode_passation",      "source": "app_fd_param_pass_march",
     "key": "id", "label": "c_mode_passation_marche",   "fact_fk": "c_mode_passation_marche"},
    {"target": "dim_tutelle_sectorielle", "source": "app_fd_param_tut_sec_ppp",
     "key": "id", "label": "c_tutelle_sectorielle",     "fact_fk": "c_tutelle_sectorielle"},
    {"target": "dim_zonage",              "source": "app_fd_param_zonage_ppp",
     "key": "id", "label": "c_zonage",                  "fact_fk": "c_zonage"},
    {"target": "dim_donateur",            "source": "app_fd_param_donateur",
     "key": "id", "label": "c_donateur",                "fact_fk": "c_donateur"},
    {"target": "dim_situation",           "source": "app_fd_param_situation_pj",
     "key": "id", "label": "c_situation_actu_projet",   "fact_fk": "c_situation_actuelle_projet"},

    # ---- Geographic dimensions (integer keys) -------------------------------
    {"target": "dim_gouvernorat", "source": "app_fd_gouvernorat",
     "key": "id", "label": "c_lib_gouv_fr", "label_ar": "c_lib_gouv",
     "fact_fk": "c_gouvernorat"},
    {"target": "dim_delegation",  "source": "app_fd_delegation",
     "key": "id", "label": "c_lib_del_fr", "label_ar": "c_lib_del",
     "parent": ("c_id_gouv", "gouvernorat_key"),       "fact_fk": "c_delegation"},
    {"target": "dim_secteur",     "source": "app_fd_secteur",
     "key": "id", "label": "c_lib_secteur_fr", "label_ar": "c_lib_secteur",
     "parent": ("c_id_del", "delegation_key"),         "fact_fk": "c_commune"},

    # ---- PGD / PGD-Etude reference dimensions (label-keyed) -----------------
    {"target": "dim_domaine_pgd",   "source": "app_fd_param_domaine_pgd",
     "key": "id", "label": "c_domaine",   "fact_fk": None},
    {"target": "dim_theme_pgd",     "source": "app_fd_param_theme_pgd",
     "key": "id", "label": "c_theme",     "fact_fk": None},
    {"target": "dim_categorie_pgd", "source": "app_fd_param_categorie_pgd",
     "key": "id", "label": "c_categorie", "fact_fk": None},
    {"target": "dim_bailleur_pgd",  "source": "app_fd_param_bailleur_pgd",
     "key": "id", "label": "c_bailleur",  "fact_fk": None},
]


def _clean_label(s: pd.Series) -> pd.Series:
    """Normalise smart quotes / dashes that break MySQL latin imports."""
    return (s.astype(str)
             .str.replace(r"[‘’]", "'", regex=True)
             .str.replace(r"[“”]", '"', regex=True)
             .str.replace(r"[–—]", "-", regex=True)
             .str.strip())


def build_dimensions(src_engine) -> tuple[dict, dict]:
    """
    Read every dimension source from jwdb and return:
      dim_frames : {dim_table_name: cleaned DataFrame}
      label_maps : {fact_fk_column: {key -> label}}  (used to denormalise the fact)
    """
    dim_frames, label_maps = {}, {}

    with src_engine.connect() as conn:
        for spec in DIM_SPECS:
            try:
                df = pd.read_sql(f"SELECT * FROM {spec['source']}", conn)
            except Exception as e:
                print(f"  WARN dim {spec['target']}: cannot read {spec['source']} ({e})")
                continue

            out = pd.DataFrame()
            out["key"] = df[spec["key"]].astype(str).str.strip()
            out["label"] = _clean_label(df[spec["label"]]) if spec["label"] in df.columns else ""

            if spec.get("label_ar") and spec["label_ar"] in df.columns:
                out["label_ar"] = _clean_label(df[spec["label_ar"]])

            if spec.get("parent"):
                src_col, dim_col = spec["parent"]
                if src_col in df.columns:
                    out[dim_col] = df[src_col].astype(str).str.strip()

            out = out[out["key"].notna() & (out["key"] != "") & (out["key"].str.lower() != "none")]
            out = out.drop_duplicates(subset="key").reset_index(drop=True)

            dim_frames[spec["target"]] = out

            if spec.get("fact_fk"):
                label_maps[spec["fact_fk"]] = dict(zip(out["key"], out["label"]))

            print(f"  dim {spec['target']:<26} <- {spec['source']:<28} ({len(out)} rows)")

    return dim_frames, label_maps
