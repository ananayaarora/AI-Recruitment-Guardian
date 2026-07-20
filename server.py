"""
server.py — Flask backend for AI Recruitment Guardian
=========================================================
Run from the project root with:
    python server.py

Expects best_model.pkl, tfidf_vectorizer.pkl, onehot_encoder.pkl
at the project root (same folder as this file).
"""

from flask import Flask, render_template, request, jsonify
from ml_pipeline import GuardianModel, scrape_job_posting

app = Flask(__name__)
guardian = GuardianModel()


@app.route("/")
def index():
    return render_template("index.html", categories=guardian.category_options)


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(force=True)
    job = {
        "title": data.get("title", ""),
        "company_profile": data.get("company_profile", ""),
        "description": data.get("description", ""),
        "requirements": data.get("requirements", ""),
        "benefits": data.get("benefits", ""),
        "employment_type": data.get("employment_type", "Not Specified"),
        "required_experience": data.get("required_experience", "Not Specified"),
        "required_education": data.get("required_education", "Not Specified"),
        "industry": data.get("industry", "Not Specified"),
        "function": data.get("function", "Not Specified"),
        "telecommuting": int(bool(data.get("telecommuting"))),
        "has_company_logo": int(bool(data.get("has_company_logo"))),
        "has_questions": int(bool(data.get("has_questions"))),
        "has_salary": int(bool(data.get("has_salary"))),
        "has_department": int(bool(data.get("has_department"))),
    }
    if not job["title"] and not job["description"]:
        return jsonify({"error": "Please provide at least a job title or description."}), 400

    result = guardian.predict_and_explain(job)
    return jsonify(result)


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Please provide a URL."}), 400

    try:
        scraped = scrape_job_posting(url)
    except Exception as e:
        return jsonify({"error": f"Could not fetch/parse this URL ({e}). "
                                  f"This site may need JavaScript or block scraping — "
                                  f"try pasting the details manually instead."}), 502

    job = {"title": scraped["title"], "description": scraped["description"],
           "requirements": scraped["requirements"]}
    result = guardian.predict_and_explain(job)
    result["scraped_title"] = scraped["title"]
    result["scraped_word_count"] = scraped["raw_word_count"]
    result["scraped_description"] = scraped["description"]
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
