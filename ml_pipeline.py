"""
ml_pipeline.py — Core ML logic for AI Recruitment Guardian
=============================================================
Shared by server.py (Flask). Identical logic to 04_Explainable_AI.ipynb
Sections 3-6, kept framework-agnostic so it can be reused anywhere.
"""

import re
import string
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix

SUSPICIOUS_PHRASES = [
    "registration fee", "no experience required", "immediate joining",
    "no interview", "limited seats", "earn money fast", "work from home",
    "quick money", "easy money", "wire transfer", "processing fee"
]

CATEGORICAL_COLS = ["employment_type", "required_experience",
                     "required_education", "industry", "function"]

DEFAULT_JOB_FIELDS = {
    "title": "", "company_profile": "", "description": "", "requirements": "", "benefits": "",
    "employment_type": "Not Specified", "required_experience": "Not Specified",
    "required_education": "Not Specified", "industry": "Not Specified", "function": "Not Specified",
    "telecommuting": 0, "has_company_logo": 0, "has_questions": 0,
    "has_salary": 0, "has_department": 0,
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class GuardianModel:
    """Loads artifacts once and exposes predict_and_explain()."""

    def __init__(self, model_path="best_model.pkl", tfidf_path="tfidf_vectorizer.pkl",
                 ohe_path="onehot_encoder.pkl"):
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(ohe_path, "rb") as f:
            self.ohe = pickle.load(f)

        self.model = model_obj["model"]
        self.feature_names = model_obj["feature_names"]
        self.category_options = {
            col: list(cats) for col, cats in zip(CATEGORICAL_COLS, self.ohe.categories_)
        }

        import shap
        import shap.explainers._tree as shap_tree

        _original_decode = shap_tree.decode_ubjson_buffer

        def _patched_decode(fd):
            jmodel = _original_decode(fd)
            try:
                param = jmodel["learner"]["learner_model_param"]
                bs = param["base_score"]
                if isinstance(bs, str) and bs.strip().startswith("["):
                    param["base_score"] = str(float(bs.strip("[]")))
            except (KeyError, TypeError):
                pass
            return jmodel

        shap_tree.decode_ubjson_buffer = _patched_decode
        self.explainer = shap.TreeExplainer(self.model)

    # ---- feature engineering ----
    def _build_feature_vector(self, job):
        job = {**DEFAULT_JOB_FIELDS, **job}

        title_clean = clean_text(job["title"])
        cp_clean = clean_text(job["company_profile"])
        desc_clean = clean_text(job["description"])
        req_clean = clean_text(job["requirements"])
        ben_clean = clean_text(job["benefits"])
        full_text = f"{title_clean} {cp_clean} {desc_clean} {req_clean} {ben_clean}"

        raw_combined_lower = f"{job['title']} {job['description']} {job['requirements']}".lower()
        flag_vals = [int(p in raw_combined_lower) for p in SUSPICIOUS_PHRASES]
        suspicious_count = sum(flag_vals)
        desc_len = len(desc_clean.split())
        has_gmail = int("gmail.com" in job["description"].lower())

        X_text = self.tfidf.transform([full_text])
        X_cat = self.ohe.transform([[job[c] for c in CATEGORICAL_COLS]])

        numeric_vals = ([job["telecommuting"], job["has_company_logo"], job["has_questions"],
                          job["has_salary"], job["has_department"], suspicious_count,
                          desc_len, has_gmail] + flag_vals)
        X_numeric = csr_matrix(np.array(numeric_vals, dtype=float).reshape(1, -1))

        X_new = hstack([X_text, X_cat, X_numeric]).tocsr()
        return X_new, job

    @staticmethod
    def _compute_trust_score(fake_probability):
        trust_score = round((1 - fake_probability) * 100, 1)
        if trust_score >= 70:
            risk_level = "low"
        elif trust_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "high"
        return trust_score, risk_level

    def _get_top_reasons(self, shap_vals_flat, top_n=6, direction="fake"):
        if direction == "fake":
            idx = np.where(shap_vals_flat > 0)[0]
        else:
            idx = np.where(shap_vals_flat < 0)[0]
        idx = idx[np.argsort(np.abs(shap_vals_flat[idx]))[::-1][:top_n]]
        reasons = []
        for i in idx:
            fname = self.feature_names[i]
            if fname.startswith("flag_"):
                kind, label = "flag", fname.replace("flag_", "").replace("_", " ")
            elif fname.startswith(tuple(c + "_" for c in CATEGORICAL_COLS)):
                kind, label = "category", fname.replace("_", " ")
            else:
                kind, label = "word", fname
            reasons.append({"kind": kind, "label": label, "shap": round(float(shap_vals_flat[i]), 4)})
        return reasons

    def predict_and_explain(self, job, top_n=6):
        X_new, job_full = self._build_feature_vector(job)

        proba_fake = float(self.model.predict_proba(X_new)[0, 1])
        prediction = "FAKE" if proba_fake >= 0.5 else "GENUINE"
        trust_score, risk_level = self._compute_trust_score(proba_fake)

        sv = self.explainer(X_new)
        vals = sv[0].values

        top_fake_reasons = self._get_top_reasons(vals, top_n=top_n, direction="fake")
        top_genuine_reasons = self._get_top_reasons(vals, top_n=top_n, direction="genuine")

        raw_combined_lower = f"{job_full['title']} {job_full['description']} {job_full['requirements']}".lower()
        active_flags = [p for p in SUSPICIOUS_PHRASES if p in raw_combined_lower]

        highlighted_html = self._highlight_text(job_full["description"] or job_full["title"], vals)

        return {
            "prediction": prediction,
            "fake_probability": round(proba_fake, 4),
            "trust_score": trust_score,
            "risk_level": risk_level,
            "top_fake_reasons": top_fake_reasons,
            "top_genuine_reasons": top_genuine_reasons,
            "active_suspicious_phrases": active_flags,
            "highlighted_html": highlighted_html,
        }

    def _highlight_text(self, text, shap_vals_flat):
        word_shap = {}
        for idx, fname in enumerate(self.feature_names):
            if fname in self.tfidf.vocabulary_:
                word_shap[fname] = shap_vals_flat[idx]

        words = text.split()
        out = []
        for w in words:
            w_clean = re.sub(r"[^\w\s]", "", w.lower())
            if w_clean in word_shap:
                val = word_shap[w_clean]
                if val > 0.01:
                    out.append(f'<mark class="hl-fake" title="SHAP {val:+.3f}">{w}</mark>')
                elif val < -0.01:
                    out.append(f'<mark class="hl-genuine" title="SHAP {val:+.3f}">{w}</mark>')
                else:
                    out.append(w)
            else:
                out.append(w)
        return " ".join(out)


def scrape_job_posting(url, timeout=10):
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")

    candidates = soup.find_all(["p", "div", "li"])
    best_block, best_len = "", 0
    for c in candidates:
        text = c.get_text(separator=" ", strip=True)
        if len(text) > best_len:
            best_block, best_len = text, len(text)

    body_text = soup.get_text(separator=" ", strip=True)
    description = best_block if best_len > 200 else body_text

    return {"title": title, "description": description, "requirements": "",
            "raw_word_count": len(description.split())}
