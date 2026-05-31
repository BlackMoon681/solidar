"""
tests/test_security.py
-----------------------
Vérifications de sécurité : isolation des secrets, CORS, validation des entrées,
protection contre les injections SQL, configuration HTTPS.

Run:  python tests/test_security.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

ROOT = Path(__file__).parent.parent

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"

_results = []

def check(name, condition, detail="", warn=False):
    marker = (WARN if warn else FAIL) if not condition else PASS
    line   = f"[{marker}]  {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    _results.append((name, condition or warn))
    return condition


# ---------------------------------------------------------------------------
# SC-01 — .env non commis (absent de git history ou .gitignore)
# ---------------------------------------------------------------------------

def test_sc01_env_gitignored():
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        check("SC-01 | .env absent de git", False,
              ".gitignore manquant — risque de commit accidentel des secrets")
        return
    content = gitignore.read_text(encoding="utf-8")
    ok = ".env" in content
    check("SC-01 | .env listé dans .gitignore",
          ok, detail=".gitignore contient .env" if ok else ".env ABSENT du .gitignore")


# ---------------------------------------------------------------------------
# SC-02 — Pas de credentials en dur dans les fichiers source
# ---------------------------------------------------------------------------

def test_sc02_no_hardcoded_credentials():
    PATTERNS = [
        r'password\s*=\s*["\'][^"\']{4,}["\']',
        r'passwd\s*=\s*["\'][^"\']{4,}["\']',
        r'secret\s*=\s*["\'][^"\']{4,}["\']',
        r'api_key\s*=\s*["\'][^"\']{8,}["\']',
        r'GROQ_API_KEY\s*=\s*["\'][^"\']{8,}["\']',
    ]
    py_files = [
        p for p in ROOT.rglob("*.py")
        if ".venv" not in str(p) and ".git" not in str(p)
    ]
    violations = []
    for py in py_files:
        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
            for pat in PATTERNS:
                for m in re.finditer(pat, src, re.IGNORECASE):
                    # Ignore os.getenv() references
                    line = src[max(0, m.start()-20):m.end()+20]
                    if "getenv" not in line and "os.environ" not in line:
                        violations.append(f"{py.relative_to(ROOT)}:{m.group()[:60]}")
        except Exception:
            pass
    check("SC-02 | Aucun credential en dur dans le code source",
          len(violations) == 0,
          detail="\n         ".join(violations[:5]) if violations else "")


# ---------------------------------------------------------------------------
# SC-03 — Variables d'environnement critiques présentes dans .env
# ---------------------------------------------------------------------------

def test_sc03_env_variables():
    env_file = ROOT / ".env"
    if not env_file.exists():
        check("SC-03 | Fichier .env présent", False, ".env introuvable")
        return

    content  = env_file.read_text(encoding="utf-8")
    required = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME", "GROQ_API_KEY"]
    missing  = [k for k in required if k not in content]
    check("SC-03 | Variables critiques dans .env",
          len(missing) == 0,
          detail=f"Manquantes : {missing}" if missing else f"{len(required)} variables présentes")


# ---------------------------------------------------------------------------
# SC-04 — Paramètres CORS restreints (pas de wildcard *)
# ---------------------------------------------------------------------------

def test_sc04_cors_restricted():
    main_file = ROOT / "app" / "main.py"
    if not main_file.exists():
        check("SC-04 | CORS — origines restreintes", False, "app/main.py introuvable")
        return
    src = main_file.read_text(encoding="utf-8")

    # Check no allow_origins=["*"]
    wildcard = bool(re.search(r'allow_origins\s*=\s*\[\s*["\*]["\*]?\s*\]', src))
    localhost_only = 'allow_origins' in src and 'localhost' in src

    check("SC-04 | CORS — wildcard '*' absent",
          not wildcard,
          detail="allow_origins=['*'] détecté !" if wildcard else "Origines explicites configurées")
    check("SC-04b | CORS — origines restreintes à localhost",
          localhost_only,
          detail="allow_origins contient 'localhost'" if localhost_only else "Origines CORS non vérifiées")


# ---------------------------------------------------------------------------
# SC-05 — Protection contre injection SQL (pas de f-string avec entrée user)
# ---------------------------------------------------------------------------

def test_sc05_sql_injection():
    risky_files = list((ROOT / "etl").glob("*.py")) + \
                  list((ROOT / "app").rglob("*.py")) + \
                  [ROOT / "pipeline.py"]

    violations = []
    # Pattern: f"...{variable}..." inside execute() or read_sql() calls
    pat = re.compile(r'(?:execute|read_sql)\s*\(\s*f["\']', re.IGNORECASE)
    for f in risky_files:
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(src):
            violations.append(f"{f.relative_to(ROOT)}: {m.group()}")

    check("SC-05 | Pas d'injection SQL via f-string dans les requêtes",
          len(violations) == 0,
          detail="\n         ".join(violations) if violations else "Requêtes paramétrées ou constantes uniquement")


# ---------------------------------------------------------------------------
# SC-06 — Modèle pkl signé (vérification intégrité)
# ---------------------------------------------------------------------------

def test_sc06_model_integrity():
    import hashlib
    model_path  = ROOT / "ml" / "model" / "risk_model.pkl"
    metrics_path = ROOT / "ml" / "model" / "model_metrics.json"

    if not model_path.exists():
        check("SC-06 | Modèle pkl présent", False)
        return

    size_kb = model_path.stat().st_size / 1024
    sha256  = hashlib.sha256(model_path.read_bytes()).hexdigest()

    check("SC-06 | Modèle pkl présent et non vide",
          size_kb > 10,
          detail=f"Taille : {size_kb:.1f} Ko  SHA256 : {sha256[:16]}...")

    # Verify metrics file consistency
    if metrics_path.exists():
        import json
        m   = json.loads(metrics_path.read_text(encoding="utf-8"))
        auc = m.get("holdout_auc", 0)
        check("SC-06b | Métriques cohérentes (AUC > 0.80)",
              auc > 0.80,
              detail=f"holdout_auc = {auc:.4f}")


# ---------------------------------------------------------------------------
# SC-07 — Vérification que les fichiers sensibles sont exclus du dépôt
# ---------------------------------------------------------------------------

def test_sc07_sensitive_files_excluded():
    sensitive = [".env", "pipeline_state.json", "pipeline.log"]
    gitignore  = ROOT / ".gitignore"
    if not gitignore.exists():
        check("SC-07 | Fichiers sensibles dans .gitignore", False)
        return
    content  = gitignore.read_text(encoding="utf-8")
    missing  = [f for f in sensitive if f not in content]
    check("SC-07 | Fichiers sensibles listés dans .gitignore",
          len(missing) == 0,
          detail=f"Non listés : {missing}" if missing else f"{len(sensitive)} fichiers protégés")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Tests de Sécurité — PPP-Risk Platform")
    print("=" * 65 + "\n")

    test_sc01_env_gitignored()
    test_sc02_no_hardcoded_credentials()
    test_sc03_env_variables()
    test_sc04_cors_restricted()
    test_sc05_sql_injection()
    test_sc06_model_integrity()
    test_sc07_sensitive_files_excluded()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(f"\n{'=' * 65}")
    print(f"  Résultat : {passed}/{total} contrôles passés")
    print("=" * 65)
    sys.exit(0 if passed == total else 1)
