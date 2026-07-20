// ==========================================================
// AI Recruitment Guardian — frontend logic
// ==========================================================

const tabs = document.querySelectorAll(".tab");
const pasteForm = document.getElementById("paste-form");
const urlForm = document.getElementById("url-form");
const scanLine = document.getElementById("scan-line");

const resultEmpty = document.getElementById("result-empty");
const resultBody = document.getElementById("result-body");
const resultError = document.getElementById("result-error");

const stamp = document.getElementById("stamp");
const stampVerdict = document.getElementById("stamp-verdict");
const stampScore = document.getElementById("stamp-score");
const predLabel = document.getElementById("pred-label");
const predProba = document.getElementById("pred-proba");
const predRisk = document.getElementById("pred-risk");
const fakeList = document.getElementById("fake-reasons-list");
const genuineList = document.getElementById("genuine-reasons-list");
const flagsBlock = document.getElementById("flags-block");
const flagChips = document.getElementById("flag-chips");
const excerpt = document.getElementById("excerpt");

// ---------- Tab switching ----------
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    const target = tab.dataset.tab === "paste" ? pasteForm : urlForm;
    target.classList.add("active");
  });
});

// ---------- Helpers ----------
function setLoading(isLoading) {
  scanLine.classList.toggle("active", isLoading);
  document.querySelectorAll(".run-btn").forEach((b) => (b.disabled = isLoading));
}

function riskLabel(risk) {
  return { low: "Low risk", medium: "Medium risk", high: "High risk" }[risk] || risk;
}
function stampWord(risk) {
  return { low: "VERIFIED", medium: "CAUTION", high: "FLAGGED" }[risk] || "REVIEWED";
}

function renderReasons(listEl, reasons) {
  listEl.innerHTML = "";
  if (!reasons || reasons.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No strong signals found.";
    li.style.borderLeftColor = "var(--rule-strong)";
    listEl.appendChild(li);
    return;
  }
  reasons.forEach((r) => {
    const li = document.createElement("li");
    let prefix = "";
    if (r.kind === "flag") prefix = "Suspicious phrase — ";
    else if (r.kind === "category") prefix = "Category signal — ";
    else if (r.kind === "metadata") prefix = "Metadata — ";
    else prefix = "Wording — ";
    li.innerHTML = `${prefix}<strong>${r.label}</strong> <span class="reason-shap">SHAP ${r.shap > 0 ? "+" : ""}${r.shap}</span>`;
    listEl.appendChild(li);
  });
}

function renderResult(data) {
  resultError.hidden = true;
  resultEmpty.hidden = true;
  resultBody.hidden = false;

  stamp.classList.remove("show", "low", "medium", "high");
  void stamp.offsetWidth; // restart animation
  stamp.classList.add(data.risk_level);
  stampVerdict.textContent = stampWord(data.risk_level);
  stampScore.textContent = data.trust_score;
  requestAnimationFrame(() => stamp.classList.add("show"));

  predLabel.textContent = data.prediction;
  predProba.textContent = (data.fake_probability * 100).toFixed(1) + "%";
  predRisk.textContent = riskLabel(data.risk_level);

  renderReasons(fakeList, data.top_fake_reasons);
  renderReasons(genuineList, data.top_genuine_reasons);

  if (data.active_suspicious_phrases && data.active_suspicious_phrases.length) {
    flagsBlock.hidden = false;
    flagChips.innerHTML = "";
    data.active_suspicious_phrases.forEach((p) => {
      const chip = document.createElement("span");
      chip.className = "flag-chip";
      chip.textContent = p;
      flagChips.appendChild(chip);
    });
  } else {
    flagsBlock.hidden = true;
  }

  excerpt.innerHTML = data.highlighted_html || "<em>No text available to annotate.</em>";
}

function renderError(message) {
  resultEmpty.hidden = true;
  resultBody.hidden = true;
  resultError.hidden = false;
  resultError.textContent = message;
}

// ---------- Paste form submit ----------
pasteForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(pasteForm);
  const payload = {
    title: fd.get("title") || "",
    description: fd.get("description") || "",
    requirements: fd.get("requirements") || "",
    company_profile: fd.get("company_profile") || "",
    employment_type: fd.get("employment_type"),
    required_experience: fd.get("required_experience"),
    required_education: fd.get("required_education"),
    industry: fd.get("industry"),
    function: fd.get("function"),
    telecommuting: fd.get("telecommuting") === "on",
    has_company_logo: fd.get("has_company_logo") === "on",
    has_questions: fd.get("has_questions") === "on",
    has_salary: fd.get("has_salary") === "on",
    has_department: fd.get("has_department") === "on",
  };

  if (!payload.title && !payload.description) {
    renderError("Please enter at least a job title or description.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    renderResult(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
});

// ---------- URL form submit ----------
urlForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(urlForm);
  const url = fd.get("url");

  setLoading(true);
  try {
    const res = await fetch("/api/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not verify this URL.");
    renderResult(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
});
