# -*- coding: utf-8 -*-
import os
import json
import math
import re
import ipaddress
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import joblib
import tldextract
from urllib.parse import urlparse

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# ==========================================================
# Configuration
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

# Default folders (sesuai struktur project kamu)
DEFAULT_DATASET_PATH = BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATASET_PATH = Path(os.getenv("PHISH_DATASET_PATH", str(DEFAULT_DATASET_PATH)))
ARTIFACTS_DIR = Path(os.getenv("PHISH_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))

# Full-feature artifacts (dari notebook full-feature)
FULL_MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
FULL_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
FULL_THRESH_PATH = ARTIFACTS_DIR / "best_threshold.json"
FULL_VERSIONS_PATH = ARTIFACTS_DIR / "versions_full.json"

# URL-only artifacts (fallback, dari notebook hybrid)
URLONLY_MODEL_PATH = ARTIFACTS_DIR / "model_urlonly.joblib"
URLONLY_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema_urlonly.json"
URLONLY_THRESH_PATH = ARTIFACTS_DIR / "best_threshold_urlonly.json"
URLONLY_VERSIONS_PATH = ARTIFACTS_DIR / "versions_urlonly.json"

# Optional: feature meaning dictionaries (kalau disimpan dari notebook)
FULL_FEATURE_DICT = ARTIFACTS_DIR / "feature_dict_full.json"
URLONLY_FEATURE_DICT = ARTIFACTS_DIR / "feature_dict_urlonly.json"

PAGE_SIZE_DEFAULT = 50

# tldextract: gunakan snapshot internal (tanpa fetch network)
_TLD = tldextract.TLDExtract(suffix_list_urls=None)

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "buff.ly",
    "rebrand.ly", "ow.ly", "lnkd.in", "rb.gy"
}

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "verification", "secure", "account", "update", "confirm",
    "signin", "sign-in", "bank", "payment", "wallet", "support", "webscr", "password"
]

# ==========================================================
# Helpers
# ==========================================================
def normalize_url(u: str) -> str:
    """Normalization ringan untuk pencarian di dataset."""
    if u is None:
        return ""
    u = str(u).strip().replace(" ", "")
    u = u.lower()
    if len(u) > 1 and u.endswith("/"):
        u = u[:-1]
    return u

def candidate_lookup_keys(user_url: str) -> List[str]:
    """Generate kandidat normalisasi untuk meningkatkan kemungkinan match di dataset."""
    raw = normalize_url(user_url)
    if not raw:
        return []

    keys = set()
    keys.add(raw)

    has_scheme = bool(re.match(r"^https?://", raw))
    if not has_scheme:
        keys.add("https://" + raw)
        keys.add("http://" + raw)
    else:
        if raw.startswith("https://"):
            keys.add("http://" + raw[len("https://"):])
        if raw.startswith("http://"):
            keys.add("https://" + raw[len("http://"):])

    keys.add(raw + "/")

    def strip_www(x: str) -> str:
        return re.sub(r"^(https?://)www\.", r"\1", x)

    keys |= {strip_www(k) for k in list(keys)}
    return [k for k in keys if k]

def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def autodetect_dataset_path() -> Path:
    """Kalau path default tidak ada, cari CSV pertama di dataset/ atau data/."""
    if DATASET_PATH.exists():
        return DATASET_PATH
    candidates: List[Path] = []
    for folder in [BASE_DIR / "dataset", BASE_DIR / "data"]:
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.csv")))
    return candidates[0] if candidates else DATASET_PATH

def explain_sklearn_pickle_error(e: Exception, versions_path: Path) -> str:
    msg = str(e)
    extra = ""
    if "_RemainderColsList" in msg or "sklearn.compose._column_transformer" in msg:
        ver = safe_read_json(versions_path) or {}
        want = ver.get("scikit_learn")
        if want:
            extra = f" (Kemungkinan mismatch scikit-learn. Coba install scikit-learn=={want} sesuai {versions_path.name})"
        else:
            extra = " (Kemungkinan mismatch scikit-learn. Samakan versi scikit-learn antara notebook training dan Flask)"
    return msg + extra

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return float(ent)

def compute_urlonly_features(url: str) -> Dict[str, Any]:
    """Hitung fitur URL-only + kebaruan (tanpa crawling)."""
    u = normalize_url(url)
    parsed = urlparse(u if re.match(r"^https?://", u) else ("http://" + u))
    scheme = parsed.scheme or ""
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]

    host = netloc
    has_port = 0
    if ":" in host:
        if host.startswith("[") and "]" in host:
            h = host.split("]")[0].lstrip("[")
            rest = host.split("]")[1]
            if rest.startswith(":"):
                has_port = 1
            host = h
        else:
            parts = host.split(":")
            host = parts[0]
            has_port = 1

    ext = _TLD(host)
    tld = (ext.suffix or "").lower()
    subdomain = (ext.subdomain or "").lower()
    domain = (ext.domain or "").lower()

    url_len = len(u)
    domain_full = ".".join([p for p in [domain, tld] if p])
    domain_len = len(domain_full)

    is_https = 1 if scheme.lower() == "https" else 0

    is_domain_ip = 0
    try:
        ipaddress.ip_address(host)
        is_domain_ip = 1
    except Exception:
        is_domain_ip = 0

    n_sub = 0 if subdomain.strip() == "" else len([x for x in subdomain.split(".") if x])

    letters = sum(ch.isalpha() for ch in u)
    digits = sum(ch.isdigit() for ch in u)
    specials = max(url_len - letters - digits, 0)

    letter_ratio = letters / url_len if url_len else 0.0
    digit_ratio = digits / url_len if url_len else 0.0
    special_ratio = specials / url_len if url_len else 0.0

    lower = u.lower()
    kw_hits = 0
    kw_flags: Dict[str, int] = {}
    for kw in SUSPICIOUS_KEYWORDS:
        hit = 1 if kw in lower else 0
        kw_flags[f"kw_{re.sub(r'[^a-z0-9]+','_',kw)}"] = hit
        kw_hits += hit

    feats: Dict[str, Any] = {
        "url_length": url_len,
        "domain_length": domain_len,
        "is_https": is_https,
        "is_domain_ip": is_domain_ip,
        "tld": tld if tld else "unknown",
        "tld_length": len(tld) if tld else 0,
        "num_subdomains": n_sub,
        "has_port": has_port,
        "path_length": len(path),
        "query_length": len(query),
        "letters_count": letters,
        "digits_count": digits,
        "specials_count": specials,
        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "count_dot": u.count("."),
        "count_hyphen": u.count("-"),
        "count_at": u.count("@"),
        "count_qmark": u.count("?"),
        "count_equal": u.count("="),
        "count_ampersand": u.count("&"),
        "count_percent": u.count("%"),
        "count_slash": u.count("/"),
        "count_underscore": u.count("_"),
        "count_hash": u.count("#"),
        "has_punycode": 1 if "xn--" in u else 0,
        "url_entropy": shannon_entropy(u),
        "is_shortener": 1 if domain_full in SHORTENER_DOMAINS else 0,
        "suspicious_kw_count": kw_hits,
    }
    feats.update(kw_flags)
    return feats

# ==========================================================
# Auto-calibration helpers (fix class mapping & threshold)
# ==========================================================
def _get_classes(model) -> Optional[List[Any]]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        return list(getattr(model.named_steps["clf"], "classes_", [])) or None
    return None

def _best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    best = {"thr": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (proba >= thr).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        if f1 > best["f1"]:
            best = {
                "thr": float(thr),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(acc),
            }
    return best

def calibrate_phishing(model, X: pd.DataFrame, y_true_phish01: np.ndarray, sample_n: int = 1500) -> Optional[Dict[str, Any]]:
    """Auto-calibration untuk mapping class + threshold."""
    if model is None or X is None or len(X) == 0:
        return None
    classes = _get_classes(model)
    if not classes or len(classes) < 2:
        return None

    n = min(sample_n, len(X))
    idx = np.random.RandomState(42).choice(len(X), size=n, replace=False)
    Xs = X.iloc[idx]
    ys = y_true_phish01[idx]

    try:
        proba = model.predict_proba(Xs)
    except Exception:
        return None

    best = None
    for j, cls in enumerate(classes):
        proba_phish = proba[:, j]
        thr = _best_threshold_by_f1(ys, proba_phish)
        cand = {
            "phishing_class_value": cls,
            "class_index": int(j),
            "best_threshold": float(thr["thr"]),
            "f1": float(thr["f1"]),
            "precision": float(thr["precision"]),
            "recall": float(thr["recall"]),
            "accuracy": float(thr["accuracy"]),
            "classes_": classes,
        }
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    return best

def _prob_phishing(model, X: pd.DataFrame, calib: Optional[Dict[str, Any]]) -> float:
    proba = model.predict_proba(X)
    classes = _get_classes(model) or []
    if calib and "phishing_class_value" in calib and calib["phishing_class_value"] in classes:
        j = classes.index(calib["phishing_class_value"])
        return float(proba[:, j][0])
    if 1 in classes:
        return float(proba[:, classes.index(1)][0])
    return float(proba[:, -1][0])

# ==========================================================
# Load Assets
# ==========================================================
def load_full_assets() -> Tuple[Optional[object], Dict[str, Any]]:
    info: Dict[str, Any] = {"errors": [], "calibration": None}
    pipeline = None
    schema = safe_read_json(FULL_SCHEMA_PATH) or {}
    thr = safe_read_json(FULL_THRESH_PATH) or {}
    ver = safe_read_json(FULL_VERSIONS_PATH) or {}
    fdict = safe_read_json(FULL_FEATURE_DICT) or {}

    if FULL_MODEL_PATH.exists():
        try:
            pipeline = joblib.load(FULL_MODEL_PATH)
        except Exception as e:
            info["errors"].append("Gagal load model_pipeline.joblib: " + explain_sklearn_pickle_error(e, FULL_VERSIONS_PATH))
    else:
        info["errors"].append(f"Full model tidak ditemukan: {FULL_MODEL_PATH}")

    used_features = schema.get("used_features", [])
    if not used_features:
        info["errors"].append(f"feature_schema.json tidak ada / used_features kosong: {FULL_SCHEMA_PATH}")

    info.update({
        "best_model_name": thr.get("best_model", "unknown"),
        "threshold_value": float(thr.get("best_threshold_by_f1", 0.5)),
        "used_features": used_features,
        "versions": ver,
        "feature_dict": fdict,
    })
    return pipeline, info

def load_urlonly_assets() -> Tuple[Optional[object], Dict[str, Any]]:
    info: Dict[str, Any] = {"errors": [], "calibration": None}
    pipeline = None
    schema = safe_read_json(URLONLY_SCHEMA_PATH) or {}
    thr = safe_read_json(URLONLY_THRESH_PATH) or {}
    ver = safe_read_json(URLONLY_VERSIONS_PATH) or {}
    fdict = safe_read_json(URLONLY_FEATURE_DICT) or {}

    if URLONLY_MODEL_PATH.exists():
        try:
            pipeline = joblib.load(URLONLY_MODEL_PATH)
        except Exception as e:
            info["errors"].append("Gagal load model_urlonly.joblib: " + explain_sklearn_pickle_error(e, URLONLY_VERSIONS_PATH))
    else:
        info["errors"].append(f"URL-only model tidak ditemukan: {URLONLY_MODEL_PATH}")

    used_features = schema.get("used_features", [])
    if not used_features:
        info["errors"].append(f"feature_schema_urlonly.json tidak ada / used_features kosong: {URLONLY_SCHEMA_PATH}")

    info.update({
        "best_model_name": thr.get("best_model", "unknown"),
        "threshold_value": float(thr.get("best_threshold_by_f1", 0.5)),
        "used_features": used_features,
        "versions": ver,
        "feature_dict": fdict,
    })
    return pipeline, info

def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

    # ---------------------------
    # Load dataset
    # ---------------------------
    ds_path = autodetect_dataset_path()
    df: Optional[pd.DataFrame] = None
    ds_errors: List[str] = []

    if not ds_path.exists():
        ds_errors.append(f"Dataset CSV tidak ditemukan: {ds_path}")
    else:
        try:
            df = pd.read_csv(ds_path)
        except Exception as e:
            ds_errors.append(f"Gagal membaca dataset CSV: {e}")

    # Build URL index for dataset-match
    url_index: Dict[str, int] = {}
    if df is not None and "URL" in df.columns:
        for i, u in enumerate(df["URL"].astype(str).fillna("").tolist()):
            nu = normalize_url(u)
            if nu and nu not in url_index:
                url_index[nu] = i

    # ---------------------------
    # Load models
    # ---------------------------
    full_model, full_info = load_full_assets()
    urlonly_model, urlonly_info = load_urlonly_assets()

    # ---------------------------
    # Auto-calibration
    # Dataset label convention: 0=phishing, 1=legitimate
    # y_true_phish01: 1=phishing
    # ---------------------------
    y_true_phish: Optional[np.ndarray] = None
    if df is not None and "label" in df.columns:
        try:
            y_true_phish = (df["label"].astype(int) == 0).astype(int).to_numpy()
        except Exception:
            y_true_phish = None

    if df is not None and y_true_phish is not None and full_model is not None and full_info.get("used_features"):
        try:
            X_full = df[full_info["used_features"]]
            cal = calibrate_phishing(full_model, X_full, y_true_phish, sample_n=2000)
            if cal:
                full_info["calibration"] = cal
                full_info["threshold_value"] = float(cal["best_threshold"])
        except Exception:
            pass

    if df is not None and y_true_phish is not None and urlonly_model is not None and urlonly_info.get("used_features") and "URL" in df.columns:
        try:
            urls = df["URL"].astype(str).fillna("").tolist()
            n = min(2000, len(urls))
            idx = np.random.RandomState(42).choice(len(urls), size=n, replace=False)
            feats = [compute_urlonly_features(urls[i]) for i in idx]
            X_url = pd.DataFrame(feats)
            for col in urlonly_info["used_features"]:
                if col not in X_url.columns:
                    X_url[col] = np.nan
            X_url = X_url[urlonly_info["used_features"]]
            cal = calibrate_phishing(urlonly_model, X_url, y_true_phish[idx], sample_n=n)
            if cal:
                urlonly_info["calibration"] = cal
                urlonly_info["threshold_value"] = float(cal["best_threshold"])
        except Exception:
            pass

    info: Dict[str, Any] = {
        "dataset_path": str(ds_path),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "dataset_errors": ds_errors,
        "n_rows": int(df.shape[0]) if df is not None else None,
        "n_cols": int(df.shape[1]) if df is not None else None,
        "url_index_size": int(len(url_index)),
        "full": full_info,
        "urlonly": urlonly_info,
    }

    def ready_full() -> bool:
        return bool(df is not None and full_model is not None and full_info.get("used_features"))

    def ready_urlonly() -> bool:
        return bool(urlonly_model is not None and urlonly_info.get("used_features"))

    def predict_full(row: pd.Series) -> Dict[str, Any]:
        used = full_info["used_features"]
        x = pd.DataFrame([row[used].to_dict()])
        prob = _prob_phishing(full_model, x, full_info.get("calibration"))
        thr = float(full_info.get("threshold_value", 0.5))
        pred = int(prob >= thr)
        return {
            "mode": "dataset-match (full-feature)",
            "prob_phishing": prob,
            "threshold": thr,
            "pred_label": "phishing" if pred == 1 else "legitimate",
            "risk_level": "High" if prob >= 0.8 else ("Medium" if prob >= 0.5 else "Low"),
        }

    def predict_urlonly(url: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        feats = compute_urlonly_features(url)
        used = urlonly_info["used_features"]
        x = pd.DataFrame([{k: feats.get(k) for k in used}])
        prob = _prob_phishing(urlonly_model, x, urlonly_info.get("calibration"))
        thr = float(urlonly_info.get("threshold_value", 0.5))
        pred = int(prob >= thr)
        out = {
            "mode": "url-only (general)",
            "prob_phishing": prob,
            "threshold": thr,
            "pred_label": "phishing" if pred == 1 else "legitimate",
            "risk_level": "High" if prob >= 0.8 else ("Medium" if prob >= 0.5 else "Low"),
        }
        return out, feats

    @app.context_processor
    def inject_globals():
        return {"now_year": 2026}

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", info=info, full_ready=ready_full(), urlonly_ready=ready_urlonly())

    @app.route("/random_pick", methods=["POST"])
    def random_pick():
        if df is None or df.empty:
            flash("Dataset tidak tersedia atau kosong.", "error")
            return redirect(url_for("index"))
        sample = df.sample(n=1).iloc[0]
        url_input = str(sample.get("URL", ""))
        if not url_input:
            flash("URL tidak ditemukan di dataset.", "error")
            return redirect(url_for("index"))
        return render_template("prefill.html", url_input=url_input)

    @app.route("/predict", methods=["POST"])
    def predict():
        url_input = request.form.get("url", "").strip()
        if not url_input:
            flash("URL tidak boleh kosong.", "error")
            return redirect(url_for("index"))

        if ready_full():
            for key in candidate_lookup_keys(url_input):
                if key in url_index:
                    row = df.iloc[url_index[key]]
                    pred = predict_full(row)
                    true_label = None
                    if "label" in df.columns:
                        try:
                            y0 = int(row["label"])
                            true_label = "phishing" if y0 == 0 else "legitimate"
                        except Exception:
                            true_label = None
                    feature_preview = [(k, row.get(k, None)) for k in full_info["used_features"][:12]]
                    return render_template(
                        "result.html",
                        url_input=url_input,
                        pred=pred,
                        true_label=true_label,
                        feature_preview=feature_preview,
                        urlonly_preview=None,
                        all_urlonly=None,
                        info=info,
                    )

        if ready_urlonly():
            pred, feats = predict_urlonly(url_input)
            used = urlonly_info["used_features"]
            urlonly_preview = [(k, feats.get(k, None)) for k in used[:16]]
            all_urlonly = [(k, feats.get(k, None)) for k in used]
            return render_template(
                "result.html",
                url_input=url_input,
                pred=pred,
                true_label=None,
                feature_preview=None,
                urlonly_preview=urlonly_preview,
                all_urlonly=all_urlonly,
                info=info,
            )

        flash("Aplikasi belum siap. Pastikan dataset CSV & artifacts model sudah ditempatkan dengan benar.", "error")
        return redirect(url_for("index"))

    @app.route("/api/predict", methods=["GET", "POST"])
    def api_predict():
        url_input = (request.args.get("url") or "").strip()
        if request.is_json:
            body = request.get_json(silent=True) or {}
            url_input = (body.get("url") or url_input).strip()
        if request.form:
            url_input = (request.form.get("url") or url_input).strip()

        if not url_input:
            return jsonify({"ok": False, "error": "url_required"}), 400

        if ready_full():
            for key in candidate_lookup_keys(url_input):
                if key in url_index:
                    row = df.iloc[url_index[key]]
                    pred = predict_full(row)
                    return jsonify({"ok": True, **pred, "source": "dataset-match"}), 200

        if ready_urlonly():
            pred, feats = predict_urlonly(url_input)
            return jsonify({"ok": True, **pred, "source": "url-only", "features": feats}), 200

        return jsonify({"ok": False, "error": "models_not_ready", "detail": info}), 500

    @app.route("/features", methods=["GET"])
    def features():
        return render_template(
            "features.html",
            info=info,
            full_features=full_info.get("used_features", []),
            urlonly_features=urlonly_info.get("used_features", []),
            full_dict=full_info.get("feature_dict", {}),
            urlonly_dict=urlonly_info.get("feature_dict", {}),
        )

    @app.route("/dataset", methods=["GET"])
    def dataset():
        if df is None:
            flash("Dataset belum tersedia.", "error")
            return redirect(url_for("index"))

        q = request.args.get("q", "").strip()
        label_filter = request.args.get("label", "").strip()
        page = max(int(request.args.get("page", "1")), 1)
        page_size = max(int(request.args.get("page_size", str(PAGE_SIZE_DEFAULT))), 1)

        view = df
        if q and "URL" in view.columns:
            nq = normalize_url(q)
            view = view[view["URL"].astype(str).str.lower().str.contains(nq, na=False)]

        if label_filter in ("phishing", "legitimate") and "label" in view.columns:
            want = 0 if label_filter == "phishing" else 1
            view = view[view["label"] == want]

        total = int(view.shape[0])
        start = (page - 1) * page_size
        end = start + page_size
        page_df = view.iloc[start:end]

        cols: List[str] = []
        if "URL" in page_df.columns:
            cols.append("URL")
        if "label" in page_df.columns:
            cols.append("label")
        for c in ["URLLength", "DomainLength", "IsHTTPS", "NoOfSubDomain", "URLSimilarityIndex"]:
            if c in page_df.columns and c not in cols:
                cols.append(c)
        cols = cols[:8] if cols else list(page_df.columns[:8])

        rows = [{c: r[c] for c in cols} for _, r in page_df[cols].iterrows()]
        total_pages = max((total + page_size - 1) // page_size, 1)

        return render_template(
            "dataset.html",
            info=info,
            rows=rows,
            cols=cols,
            q=q,
            label_filter=label_filter,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    @app.route("/model", methods=["GET"])
    def model():
        return render_template("model.html", info=info, full_ready=ready_full(), urlonly_ready=ready_urlonly())

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "dataset_loaded": df is not None,
            "full_ready": ready_full(),
            "urlonly_ready": ready_urlonly(),
            "calibration": {
                "full": full_info.get("calibration"),
                "urlonly": urlonly_info.get("calibration"),
            },
            "errors": {
                "dataset": info.get("dataset_errors", []),
                "full": info.get("full", {}).get("errors", []),
                "urlonly": info.get("urlonly", {}).get("errors", []),
            },
        })

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
