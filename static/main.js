// =====================================================================
// static/main.js
// ---------------------------------------------------------------------
// Frontend controller script for document ingestion and LLM query flow.
// ---------------------------------------------------------------------
// Responsibilities are divided cleanly:
//   • Handles file selection, validation, and upload to backend (/api/ingest)
//   • Manages user interactions for query input and LLM response display
//   • Renders LLM output and supporting passages with rich formatting
//
// Design Principles:
//   - Modular and self-contained (no global namespace leakage).
//   - Clear separation of UI handling and network communication.
//   - Resilient user feedback with inline spinners, color cues, and validation.
//   - Aligned with SOLID principles for extensibility and readability.
// =====================================================================

document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------------
  // DOM ELEMENT REFERENCES
  // -------------------------------------------------------------------
  const ingestForm = document.getElementById("ingest-form");
  const filesInput = document.getElementById("files");
  const ingestStatus = document.getElementById("ingest-status");
  const ingestButton = ingestForm?.querySelector('button[type="submit"]');

  const questionInput = document.getElementById("question");
  const answerArea = document.getElementById("answer-area");
  const answerBox = document.getElementById("answer-box");
  const passagesDiv = document.getElementById("passages");

  if (!ingestForm || !filesInput || !ingestStatus || !questionInput) {
    console.error("main.js: required DOM elements missing");
    return;
  }

  const MAX_FILES = 3;

  // ===================================================================
  // Utility Functions
  // ===================================================================

  /** Clear all child elements from a target element. */
  const clearElement = (el) => { if (el) el.innerHTML = ""; };

  /** Show a user message with icon and color (info, success, or error). */
  const showMessage = (el, message, type = "info") => {
    const icons = { info: "ℹ️", success: "✅", error: "❌" };
    const colors = { info: "text-yellow-300", success: "text-green-400", error: "text-red-400" };
    if (!el) return;
    el.innerHTML = `<span class="${colors[type] || colors.info}">${icons[type] || ""}</span> <span class="ml-2">${message}</span>`;
  };

  /** Create a simple CSS spinner element for async states. */
  const createSpinner = (sizePx = 16) => {
    const div = document.createElement("div");
    div.style.width = div.style.height = `${sizePx}px`;
    div.style.borderWidth = "3px";
    div.style.borderStyle = "solid";
    div.style.borderTopColor = "transparent";
    div.style.borderRightColor = "#f59e0b";
    div.style.borderBottomColor = "#f59e0b";
    div.style.borderLeftColor = "#f59e0b";
    div.style.borderRadius = "50%";
    div.style.marginRight = "8px";
    div.style.display = "inline-block";
    div.style.animation = "spin 1s linear infinite";
    const style = document.createElement("style");
    style.innerHTML = "@keyframes spin { from { transform: rotate(0deg);} to { transform: rotate(360deg); } }";
    document.head.appendChild(style);
    return div;
  };

  /** Enable or disable the ingest button, reflecting busy state. */
  const setIngestButtonState = (disabled = false) => {
    if (!ingestButton) return;
    ingestButton.disabled = disabled;
    if (disabled) {
      ingestButton.classList.add("opacity-60", "cursor-not-allowed");
      ingestButton.setAttribute("aria-busy", "true");
    } else {
      ingestButton.classList.remove("opacity-60", "cursor-not-allowed");
      ingestButton.removeAttribute("aria-busy");
    }
  };

  // ===================================================================
  // File Input Handling
  // ===================================================================

  /**
   * Restrict number of selected files and provide inline feedback.
   * Automatically truncates extra files beyond the allowed maximum.
   */
  const limitFiles = (input, maxFiles = MAX_FILES) => {
    if (!input || !input.files) return;
    if (input.files.length > maxFiles) {
      showMessage(ingestStatus, `You can select at most ${maxFiles} files. Extra files removed.`, "error");
      const dt = new DataTransfer();
      Array.from(input.files).slice(0, maxFiles).forEach(file => dt.items.add(file));
      input.files = dt.files;
    } else {
      if (input.files.length === 0) showMessage(ingestStatus, "No files selected.", "info");
      else if (input.files.length === 1) showMessage(ingestStatus, `1 file selected: ${input.files[0].name}`, "info");
      else showMessage(ingestStatus, `${input.files.length} files selected`, "info");
    }
  };

  filesInput.addEventListener("change", () => limitFiles(filesInput, MAX_FILES));

  // ===================================================================
  // File Upload & Ingestion
  // ===================================================================

  /**
   * Handle form submission for ingesting files.
   * Sends files to `/api/ingest` via POST and reports success/failure.
   */
  const handleIngest = async (e) => {
    e.preventDefault();
    const files = filesInput.files;
    if (!files || files.length === 0) {
      showMessage(ingestStatus, "No files selected. Please select at least 1 file.", "error");
      return;
    }
    if (files.length > MAX_FILES) {
      showMessage(ingestStatus, `You can select at most ${MAX_FILES} files.`, "error");
      return;
    }

    const fd = new FormData();
    for (const file of files) fd.append("files", file);

    setIngestButtonState(true);
    clearElement(ingestStatus);
    const spinner = createSpinner();
    ingestStatus.appendChild(spinner);
    ingestStatus.appendChild(document.createTextNode("Uploading..."));

    try {
      const resp = await fetch("/api/ingest", { method: "POST", body: fd });
      let data = null;
      try { data = await resp.json(); } catch (err) { console.warn("Invalid JSON", err); }

      clearElement(ingestStatus);
      if (resp.ok) {
        const added = data?.added_files || [];
        const failed = data?.failed_files || [];
        if (added.length) showMessage(ingestStatus, `Added: ${added.join(", ")}`, "success");
        else showMessage(ingestStatus, "Upload finished but no files added.", "error");
        if (failed.length) {
          const warn = document.createElement("div");
          warn.className = "text-sm text-yellow-600 mt-1";
          warn.textContent = `Failed to process: ${failed.join(", ")}`;
          ingestStatus.appendChild(warn);
        }
        filesInput.value = "";
      } else {
        const errMsg = data?.error || `${resp.status} ${resp.statusText}`;
        showMessage(ingestStatus, `Ingest failed: ${errMsg}`, "error");
      }
    } catch (err) {
      console.error("Upload error:", err);
      clearElement(ingestStatus);
      showMessage(ingestStatus, "Upload failed — network/server error.", "error");
    } finally {
      setIngestButtonState(false);
    }
  };

  ingestForm.addEventListener("submit", handleIngest);

  // ===================================================================
  // LLM Answer Rendering (Markdown + Highlighting)
  // ===================================================================

  const inlineHighlightFields = ["city", "state", "college", "role", "years_experience", "primary_language"];
  const codeFields = [
    "github_public_repos", "stackoverflow_score", "sandbox_depositor", "is_live_trader",
    "backtests_run", "strategy_saves", "strategy_publishes", "live_trades_count", "total_pnl_inr",
    "nifty_1y_return_pct", "monthly_NSE_retail_accounts"
  ];

  /**
   * Append formatted LLM-generated text to the answer area.
   * Supports basic markdown (bold, italic, list) and field-based highlights.
   */
  const appendAnswerText = (text) => {
    const div = document.createElement("div");
    div.className = "text-gray-800 font-medium leading-relaxed";

    if (!text) {
      answerBox.appendChild(div);
      return;
    }

    let html = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>')
      .replace(/\n\n/g, '<p></p>');

    if (html.includes('<li>')) html = `<ul class="list-disc ml-5 mb-3">${html}</ul>`;

    inlineHighlightFields.forEach(f => {
      const regex = new RegExp(`\\b(${f})\\b`, "gi");
      html = html.replace(regex, `<span class="font-semibold text-blue-600">$1</span>`);
    });

    codeFields.forEach(f => {
      const regex = new RegExp(`\\b(${f})\\b`, "gi");
      html = html.replace(regex, `<code class="bg-gray-100 text-green-700 px-1 rounded">$1</code>`);
    });

    div.innerHTML = html;
    answerBox.appendChild(div);
  };

  // ===================================================================
  // Supporting Passages Rendering
  // ===================================================================

  const escapeHtml = (unsafe) => unsafe
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');

  /**
   * Render a retrieved passage with highlighting and clean layout.
   */
  const appendPassage = (passage) => {
    const div = document.createElement("div");
    div.className = "passage";
    const src = passage.source || "unknown";
    const txt = passage.text || "";

    let html = txt.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    inlineHighlightFields.forEach(f => {
      const regex = new RegExp(`\\b(${f})\\b`, "gi");
      html = html.replace(regex, `<span class="font-semibold text-blue-600">$1</span>`);
    });
    codeFields.forEach(f => {
      const regex = new RegExp(`\\b(${f})\\b`, "gi");
      html = html.replace(regex, `<code class="bg-gray-100 text-green-700 px-1 rounded">$1</code>`);
    });

    const paras = html.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
    const parasHtml = paras.map(p => `<p class="mb-2 leading-relaxed">${p}</p>`).join("");

    div.innerHTML = `
      <div class="flex items-start">
        <span class="source mr-2 font-bold">${escapeHtml(src)}</span>
        <div class="flex-1">${parasHtml}</div>
      </div>`;
    passagesDiv.appendChild(div);
  };

  // ===================================================================
  // Query Submission Handling
  // ===================================================================

  /**
   * Submit a user question to the backend (/api/query) and display results.
   */
  const handleQuestion = async () => {
    const query = questionInput.value.trim();
    if (!query) {
      showMessage(answerBox, "Please type a question before submitting.", "error");
      return;
    }

    answerArea.classList.remove("hidden");
    clearElement(answerBox);
    clearElement(passagesDiv);
    answerBox.innerHTML = `<div class="text-sm text-gray-600">Thinking...</div>`;

    try {
      const resp = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 4 })
      });

      let data = null;
      try { data = await resp.json(); } catch (err) { console.warn("Invalid JSON", err); }

      clearElement(answerBox);
      if (!resp.ok) {
        const errMsg = data?.error || `${resp.status} ${resp.statusText}`;
        showMessage(answerBox, errMsg, "error");
        return;
      }

      appendAnswerText(data?.answer || "No answer generated.");
      const retrieved = data?.retrieved || [];

      if (retrieved.length === 0) {
        const hint = document.createElement("div");
        hint.className = "text-sm text-gray-600 mt-2";
        hint.textContent = "No supporting passages were returned.";
        answerBox.appendChild(hint);
      } else {
        retrieved.forEach(appendPassage);
      }

      questionInput.value = "";
      questionInput.style.height = "auto";
    } catch (err) {
      console.error("Query error:", err);
      clearElement(answerBox);
      showMessage(answerBox, "Error fetching answer — network/server error.", "error");
    }
  };

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuestion();
    }
  });

  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 400)}px`;
  });

  // ===================================================================
  // Initialization
  // ===================================================================
  limitFiles(filesInput, MAX_FILES);
});