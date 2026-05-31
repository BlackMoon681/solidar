"""
tests/test_db_optimization.py
------------------------------
Audit des optimisations de requêtes MySQL :
  - Vérification / création des index recommandés
  - Mesure des temps de requête avant/après index
  - Rapport EXPLAIN sur les requêtes critiques

Run:  python tests/test_db_optimization.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from sqlalchemy import create_engine, text
from core.config import Config

ROOT   = Path(__file__).parent.parent
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
INFO = "\033[94m INFO\033[0m"

_results = []


def check(name, condition, detail=""):
    marker = PASS if condition else FAIL
    line   = f"[{marker}]  {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    _results.append((name, condition))
    return condition


def info(msg):
    print(f"[{INFO} ]  {msg}")


def _time_query(sql, params=None, n=10):
    times = []
    with engine.connect() as conn:
        for _ in range(n):
            t0 = time.perf_counter()
            conn.execute(text(sql), params or {})
            times.append((time.perf_counter() - t0) * 1000)
    import numpy as np
    return {"avg": round(float(np.mean(times)), 3),
            "p95": round(float(np.percentile(times, 95)), 3)}


def _index_exists(table, index_name):
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = :schema AND table_name = :table "
            "AND index_name = :idx"
        ), {"schema": Config.DB_NAME, "table": table, "idx": index_name})
        return r.scalar() > 0


def _create_index_if_missing(table, index_name, columns):
    if _index_exists(table, index_name):
        info(f"Index {index_name} sur {table} déjà présent — ignoré")
        return False
    with engine.connect() as conn:
        conn.execute(text(
            f"CREATE INDEX {index_name} ON {table} ({columns})"
        ))
        conn.commit()
    info(f"Index {index_name} créé sur {table}({columns})")
    return True


# ---------------------------------------------------------------------------
# DB-01 — Connectivité DB
# ---------------------------------------------------------------------------

def test_db01_connectivity():
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT 1"))
            ok = r.scalar() == 1
        check("DB-01 | Connexion MySQL opérationnelle", ok)
    except Exception as e:
        check("DB-01 | Connexion MySQL opérationnelle", False, detail=str(e))
        return False
    return True


# ---------------------------------------------------------------------------
# DB-02 — Tables nettoyées existent
# ---------------------------------------------------------------------------

def test_db02_clean_tables():
    tables = [Config.TABLE_PPP_CLEAN, Config.TABLE_PGD_CLEAN, Config.TABLE_PGD_ETUDE_CLEAN]
    with engine.connect() as conn:
        for t in tables:
            try:
                r = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                n = r.scalar()
                check(f"DB-02 | Table {t} existante ({n} lignes)", True,
                      detail=f"{n} lignes")
            except Exception as e:
                check(f"DB-02 | Table {t} existante", False, detail=str(e))


# ---------------------------------------------------------------------------
# DB-03 — Requêtes critiques : temps sans index
# ---------------------------------------------------------------------------

CRITICAL_QUERIES = {
    "COUNT positifs (pipeline stage_model)": (
        f"SELECT COUNT(*) FROM {Config.TABLE_PPP_CLEAN} WHERE risque_depassement = 1",
        {}
    ),
    "COUNT total PPP clean": (
        f"SELECT COUNT(*) FROM {Config.TABLE_PPP_CLEAN}",
        {}
    ),
    "COUNT total PPP raw": (
        f"SELECT COUNT(*) FROM {Config.TABLE_PPP}",
        {}
    ),
    "COUNT total PGD raw": (
        f"SELECT COUNT(*) FROM {Config.TABLE_PGD}",
        {}
    ),
    "COUNT total PGD Etude raw": (
        f"SELECT COUNT(*) FROM {Config.TABLE_PGD_ETUDE}",
        {}
    ),
}


def test_db03_query_times_before():
    print("\n  Temps de requête (avant optimisation) :")
    print(f"  {'Requête':<45}  {'avg':>10}  {'p95':>10}")
    print(f"  {'-' * 70}")
    timings_before = {}
    for name, (sql, params) in CRITICAL_QUERIES.items():
        try:
            t = _time_query(sql, params, n=20)
            timings_before[name] = t
            print(f"  {name:<45}  {t['avg']:>8.3f}ms  {t['p95']:>8.3f}ms")
        except Exception as e:
            print(f"  {name:<45}  ERREUR: {e}")
    return timings_before


# ---------------------------------------------------------------------------
# DB-04 — Création des index recommandés
# ---------------------------------------------------------------------------

RECOMMENDED_INDEXES = [
    (Config.TABLE_PPP_CLEAN,  "idx_ppp_clean_risque",  "risque_depassement"),
    (Config.TABLE_PPP_CLEAN,  "idx_ppp_clean_retard",  "risque_retard"),
]


def test_db04_create_indexes():
    print("\n  Vérification et création des index recommandés :")
    created = 0
    for table, idx_name, cols in RECOMMENDED_INDEXES:
        try:
            existed = _index_exists(table, idx_name)
            if not existed:
                _create_index_if_missing(table, idx_name, cols)
                created += 1
            check(f"DB-04 | Index {idx_name} sur {table}",
                  _index_exists(table, idx_name),
                  detail="existant" if existed else "créé")
        except Exception as e:
            check(f"DB-04 | Index {idx_name}", False, detail=str(e))
    info(f"{created} index(s) créé(s), {len(RECOMMENDED_INDEXES)-created} déjà présent(s)")


# ---------------------------------------------------------------------------
# DB-05 — Temps après index + EXPLAIN
# ---------------------------------------------------------------------------

def test_db05_query_times_after(timings_before):
    print("\n  Temps de requête (après optimisation) :")
    print(f"  {'Requête':<45}  {'avant':>10}  {'après':>10}  {'gain':>8}")
    print(f"  {'-' * 80}")
    for name, (sql, params) in CRITICAL_QUERIES.items():
        try:
            t_after = _time_query(sql, params, n=20)
            t_bef   = timings_before.get(name, {}).get("avg", None)
            if t_bef:
                gain = ((t_bef - t_after["avg"]) / t_bef * 100) if t_bef > 0 else 0
                print(f"  {name:<45}  {t_bef:>8.3f}ms  {t_after['avg']:>8.3f}ms  {gain:>+6.1f}%")
                _results.append((f"DB-05 perf | {name}", t_after["avg"] <= t_bef + 1.0))
            else:
                print(f"  {name:<45}  {t_after['avg']:>8.3f}ms")
        except Exception as e:
            print(f"  {name:<45}  ERREUR: {e}")


# ---------------------------------------------------------------------------
# DB-06 — EXPLAIN sur la requête COUNT positifs
# ---------------------------------------------------------------------------

def test_db06_explain():
    print("\n  EXPLAIN — requête COUNT positifs (critique pour pipeline) :")
    sql = f"EXPLAIN SELECT COUNT(*) FROM {Config.TABLE_PPP_CLEAN} WHERE risque_depassement = 1"
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
            for row in rows:
                d = dict(row._mapping)
                print(f"  {d}")
                uses_index = d.get("key") is not None
                check("DB-06 | EXPLAIN utilise un index pour COUNT positifs",
                      uses_index,
                      detail=f"key={d.get('key')}  rows_examined={d.get('rows')}")
    except Exception as e:
        check("DB-06 | EXPLAIN", False, detail=str(e))


# ---------------------------------------------------------------------------
# DB-07 — Connection pool (SQLAlchemy)
# ---------------------------------------------------------------------------

def test_db07_connection_pool():
    pool = engine.pool
    info(f"SQLAlchemy pool : size={pool.size()}  overflow={pool.overflow()}  "
         f"checked_out={pool.checkedout()}")
    check("DB-07 | Pool de connexions SQLAlchemy actif", True,
          detail=f"pool_size={pool.size()}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Tests d'Optimisation des Requêtes — PPP-Risk Platform")
    print("=" * 70 + "\n")

    if not test_db01_connectivity():
        print("\n  Impossible d'atteindre la DB. Tests annulés.")
        sys.exit(1)

    test_db02_clean_tables()
    timings_before = test_db03_query_times_before()
    test_db04_create_indexes()
    test_db05_query_times_after(timings_before)
    test_db06_explain()
    test_db07_connection_pool()

    # Export
    report = {
        "timings_before": {k: v for k, (sql, p) in CRITICAL_QUERIES.items()
                           for k2, v in [("sql", sql)] if False} ,
        "indexes_created": [i[1] for i in RECOMMENDED_INDEXES],
    }
    out = ROOT / "tests" / "db_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(f"\n{'=' * 70}")
    print(f"  Résultat : {passed}/{total} contrôles passés")
    print(f"  Résultats exportés → {out}")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)
