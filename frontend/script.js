/**
 * InsightLens — Frontend Logic
 * Handles API calls, UI state, and dynamic result rendering.
 */

const API_BASE = "http://localhost:8000";

// --- DOM Elements ---
const reviewInput = document.getElementById("review-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const inputSection = document.getElementById("input-section");
const loadingSection = document.getElementById("loading-section");
const errorSection = document.getElementById("error-section");
const resultsSection = document.getElementById("results-section");

// --- Character Count ---
reviewInput.addEventListener("input", () => {
    const len = reviewInput.value.length;
    charCount.textContent = `${len} / 5000`;
    if (len > 5000) {
        charCount.style.color = "var(--negative)";
    } else {
        charCount.style.color = "var(--text-muted)";
    }
});

// --- Loading Step Animation ---
let stepInterval = null;

function startLoadingSteps() {
    const steps = [
        document.getElementById("step-1"),
        document.getElementById("step-2"),
        document.getElementById("step-3"),
    ];

    let current = 0;
    steps[0].classList.add("active");

    stepInterval = setInterval(() => {
        if (current < steps.length) {
            steps[current].classList.remove("active");
            steps[current].classList.add("done");
        }
        current++;
        if (current < steps.length) {
            steps[current].classList.add("active");
        }
    }, 1200);
}

function stopLoadingSteps() {
    if (stepInterval) {
        clearInterval(stepInterval);
        stepInterval = null;
    }
    // Reset step states
    document.querySelectorAll(".step").forEach((step) => {
        step.classList.remove("active", "done");
    });
}

// --- Main Analyze Function ---
async function analyzeReview() {
    const reviewText = reviewInput.value.trim();

    // Validation
    if (!reviewText) {
        shakeElement(reviewInput);
        reviewInput.focus();
        return;
    }

    if (reviewText.length < 10) {
        showError("Review Too Short", "Please enter at least 10 characters for a meaningful analysis.");
        return;
    }

    if (reviewText.length > 5000) {
        showError("Review Too Long", "Please keep your review under 5000 characters.");
        return;
    }

    // Show loading state
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    showSection("loading");
    startLoadingSteps();

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review_text: reviewText }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        stopLoadingSteps();
        renderResults(data);
        showSection("results");
    } catch (error) {
        stopLoadingSteps();
        console.error("Analysis failed:", error);

        if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
            showError(
                "Cannot Connect to Server",
                "Make sure the backend is running: uv run uvicorn nlp_cap.main:app --reload"
            );
        } else {
            showError("Analysis Failed", error.message);
        }
    } finally {
        analyzeBtn.classList.remove("loading");
        analyzeBtn.disabled = false;
    }
}

// --- Render Results ---
function renderResults(data) {
    // 1. Sentiment Badge
    const sentimentBadge = document.getElementById("sentiment-badge");
    const badgeIcon = document.getElementById("badge-icon");
    const badgeLabel = document.getElementById("badge-label");

    const sentimentLower = data.overall_sentiment.toLowerCase();
    sentimentBadge.className = `sentiment-badge ${sentimentLower}`;

    const icons = { positive: "✅", negative: "❌", neutral: "➖" };
    badgeIcon.textContent = icons[sentimentLower] || "➖";
    badgeLabel.textContent = data.overall_sentiment;

    // 2. Confidence
    const confidenceValue = document.getElementById("confidence-value");
    const confidenceFill = document.getElementById("confidence-fill");
    const pct = Math.round(data.baseline_model_confidence * 100);
    confidenceValue.textContent = `${pct}%`;

    // Animate the bar
    requestAnimationFrame(() => {
        confidenceFill.style.width = `${pct}%`;
    });

    // 3. Reasoning
    document.getElementById("reasoning-text").textContent = data.reasoning;

    // 4. Aspects
    const aspectsGrid = document.getElementById("aspects-grid");
    aspectsGrid.innerHTML = "";

    data.aspects.forEach((aspect, index) => {
        const tag = document.createElement("div");
        const aspSentiment = aspect.sentiment.toLowerCase();
        tag.className = `aspect-tag ${aspSentiment}`;
        tag.style.animationDelay = `${index * 0.08}s`;

        const aspIcons = { positive: "✅", negative: "❌", neutral: "➖" };
        tag.innerHTML = `
            <span class="aspect-icon">${aspIcons[aspSentiment] || "➖"}</span>
            <span>${capitalize(aspect.feature)}</span>
        `;
        aspectsGrid.appendChild(tag);
    });

    // 5. Recommendations
    const recList = document.getElementById("recommendations-list");
    recList.innerHTML = "";

    data.recommended_features.forEach((rec, index) => {
        const li = document.createElement("li");
        li.className = "recommendation-item";
        li.style.animationDelay = `${index * 0.12}s`;
        li.innerHTML = `
            <span class="recommendation-number">${index + 1}</span>
            <span class="recommendation-text">${rec}</span>
        `;
        recList.appendChild(li);
    });
}

// --- UI State Management ---
function showSection(section) {
    inputSection.style.display = section === "input" ? "block" : "none";
    loadingSection.style.display = section === "loading" ? "block" : "none";
    errorSection.style.display = section === "error" ? "block" : "none";
    resultsSection.style.display = section === "results" ? "flex" : "none";
}

function showError(title, message) {
    document.getElementById("error-title").textContent = title;
    document.getElementById("error-message").textContent = message;
    showSection("error");
}

function resetUI() {
    showSection("input");
    // Reset confidence bar for next animation
    document.getElementById("confidence-fill").style.width = "0%";
    reviewInput.focus();
}

// --- Utilities ---
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function shakeElement(el) {
    el.style.animation = "none";
    el.offsetHeight; // Trigger reflow
    el.style.animation = "shake 0.5s ease";
    setTimeout(() => (el.style.animation = ""), 500);
}

// --- Keyboard Shortcut ---
reviewInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        analyzeReview();
    }
});
