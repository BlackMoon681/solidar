"""
tests/run_all.py
----------------
Lance tous les tests et génère un rapport complet (console + JSON).

Run:  python tests/run_all.py
      python tests/run_all.py --skip-db   (ignore les tests DB si hors ligne)
"""
import sys, os, subprocess, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

ROOT   = Path(__file__).parent.parent
VENV   = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV) if VENV.exists() else sys.executable

SUITES = [
    ("Fonctionnels",      "tests/test_functional.py",       False),
    ("Performance",       "tests/test_performance.py",      False),
    ("Sécurité",          "tests/test_security.py",         False),
    ("Optimisation DB",   "tests/test_db_optimization.py",  True),   # db_required=True
]

SEP  = "=" * 70
SEP2 = "-" * 70


def run_suite(label, script, db_required, skip_db):
    if db_required and skip_db:
        print(f"\n  [ SKIP ]  {label} (--skip-db)")
        return {"suite": label, "status": "skipped", "duration_s": 0, "returncode": 0}

    print(f"\n{SEP}")
    print(f"  Suite : {label}")
    print(SEP)

    t0  = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    ret = subprocess.run(
        [PYTHON, str(ROOT / script)],
        env=env, cwd=str(ROOT),
    )
    elapsed = round(time.perf_counter() - t0, 2)

    status = "passed" if ret.returncode == 0 else "failed"
    return {"suite": label, "status": status,
            "duration_s": elapsed, "returncode": ret.returncode}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-db", action="store_true",
                        help="Ignore les tests nécessitant la base de données")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("  PPP-Risk Platform — Suite de Tests Complète")
    print(SEP)

    results = []
    for label, script, db_req in SUITES:
        r = run_suite(label, script, db_req, args.skip_db)
        results.append(r)

    # Summary
    print(f"\n{SEP}")
    print("  RÉSUMÉ GLOBAL")
    print(SEP)
    print(f"  {'Suite':<25}  {'Statut':<10}  {'Durée':>8}")
    print(f"  {SEP2}")
    for r in results:
        icon = "OK" if r["status"] in ("passed", "skipped") else "XX"
        print(f"  [{icon}]  {r['suite']:<23}  {r['status']:<10}  {r['duration_s']:>6.2f}s")

    n_pass = sum(1 for r in results if r["status"] == "passed")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"\n  Passés : {n_pass}  |  Ignorés : {n_skip}  |  Échoués : {n_fail}")

    out = ROOT / "tests" / "report.json"
    out.write_text(json.dumps({"suites": results,
                               "summary": {"passed": n_pass, "skipped": n_skip,
                                           "failed": n_fail}},
                              indent=2), encoding="utf-8")
    print(f"  Rapport exporte -> {out}")
    print(SEP)
    sys.exit(0 if n_fail == 0 else 1)
