# Materials Discovery Workshop: Roadmap to Crucible-Ready

## Executive Summary & Quick Start

---

## The Challenge

You've built a sophisticated materials discovery tool (VAE + ML classifier + rule-based predictor) that works well as a demo. But a metallurgical studio won't risk $500–5000 in materials and furnace time unless they trust the predictions.

**How do you move from "this looks cool" to "I'll use this to guide real experiments"?**

Answer: Three things.

1. **Replace mock data with real Materials Project data** (so predictions are grounded in known chemistry)
2. **Make uncertainty explicit and exportable** (so the studio sees exactly where they should be skeptical)
3. **Close the feedback loop** (so the model improves with their real outcomes)

---

## The Roadmap at a Glance

### Part 1: Data Realism (3–4 days)

- Swap out synthetic training data → real Materials Project alloys
- Train VAE on real feature distributions
- Enforce chemically valid compositions (sum to 1.0, 5–95% range)

### Part 2: Physics Rigor (3–4 days)

- Separate thermodynamic stability from synthesizability
- Expose configurable stability thresholds (0.025–0.1 eV/atom)
- Add process-aware recommendations (arc melt / furnace / CVD based on equipment)
- Weight-percent calculations for direct lab use

### Part 3: Uncertainty Quantification (3–4 days)

- Compute calibration metrics (is the 80% probability actually 80%?)
- Flag in-distribution vs. extrapolation candidates
- Export with full context: CSV + PDF report + model card
- Log experimental outcomes and compute hit rate

### Part 4: UX Polish (2 days)

- Model diagnostics dashboard (accuracy, precision, recall, CV scores)
- Explicit safety warnings and limitations
- Equipment selector to filter recommendations

### Part 5: Testing (3–4 days)

- Unit tests for data loading, composition validity, ML metrics
- End-to-end pipeline validation
- One real synthesis to close the loop

**Total: 4–5 weeks, ~70 hours of focused development.**

---

## Why This Matters: The Studio's Perspective

When a metallurgist sees your export CSV, they ask three questions:

1. **"Is this chemically stable?"**
   - Currently: Stability and synthesizability mixed together
   - After: Clear "Stability Score" + "E_hull" column + color codes

2. **"Can I actually make this with my equipment?"**
   - Currently: Generic "arc melting" or "furnace"
   - After: "Arc melt at 1900°C under Ar, 30 min, costs ~$180, success rate 82%"

3. **"How confident are you?"**
   - Currently: Single number (0–1), no context
   - After: ML confidence + LLM confidence + calibration flag + in-distribution status

When all three are answered clearly, they print the CSV and walk to the furnace.

---

## The Deliverables

### 📊 Day 1–5: Real Data + Export

**CSV Export: `crucible_ready_alloys_20260118.csv`**

```
| Formula | Comp1 | Comp2 | at% | wt% | Feedstock_g_100g | Stability | E_hull | Synth_Prob | ML_Conf | Method | Temp_C | Atm | Success_Prob | Est_Cost | Est_Time | Priority | In_Dist | Notes |
|---------|-------|-------|-----|-----|------------------|-----------|--------|-----------|---------|--------|--------|-----|--------------|----------|----------|----------|---------|-------|
| Al0.6Ti | 0.6   | 0.4   | 60  | 59  | 59.3g Al, 40.7g Ti | 0.92      | 0.018  | 0.87      | 0.91    | Arc M  | 1900   | Ar  | 0.84         | $180     | 2.5h     | 1        | In      | Try   |
```

### 📋 Day 5: PDF Report

**`model_report_20260118.pdf`**

- Data source: 1000 real binary alloys from Materials Project
- Model: 6→VAE→5 latent, Random Forest 200 estimators
- Performance: 87% accuracy, 0.91 precision, 0.83 recall, F1 0.87
- Limitations: Binary alloys only, thermodynamics not kinetics
- Workflow: Start with Priority 1, validate experimentally, log outcomes

### 🎛️ Week 2: Feedback Loop

**Outcomes logged in `outcomes.csv`:**

```
| Timestamp | Formula | ML_Pred | LLM_Pred | Ensemble | Actual | Notes |
|-----------|---------|---------|----------|----------|--------|-------|
| 2026-01-20 | Al0.6Ti | 0.87    | 0.92     | 0.89     | Success| Clean ingot, no cracks |
```

After 10 outcomes: "Your hit rate is 70%. Model is well-calibrated for your studio."

### 📱 Week 3: UI Enhancements

- Stability/Synth separation with color codes
- Equipment selector (Arc Melter ☑️ Furnace ☑️)
- Model Diagnostics tab with full metrics
- Outcome logging interface

---

## Three Critical Implementation Steps

### Step 1: Load Real Data (Day 1)

**File: `gradio_app.py` line ~150**

```python
from materials_discovery_api import gettrainingdataset
import os

# At startup
mp_api_key = os.getenv("MP_API_KEY", "fallback_key")
mlfeatures, rawdata = gettrainingdataset(apikey=mp_api_key, nmaterials=1000)
mlclassifier.train_on_real_data(mlfeatures)

# Add UI banner
gr.Markdown("📊 **Model trained on 1000 real binary alloys from Materials Project**")
```

### Step 2: Separate Stability from Synthesizability (Day 3)

**File: `synthesizability_predictor.py` + `gradio_app.py`**

```python
# New function
def compute_thermodynamic_stability(materials_df):
    """0-1 score: Ehull, formation energy, density, crystal complexity"""
    # Low Ehull (< 0.025) → 0.95 stability
    # High Ehull (> 0.1) → 0.2 stability
    return stability_scores

# Results table now shows:
# | Formula | Stability (green/yellow/red) | E_hull | Synth_Prob | Method | Cost | ... |
```

### Step 3: Export + Feedback (Day 5)

**File: `gradio_app.py` + new `export_for_lab.py`**

```python
# New function
def export_for_lab(results_df, model_diagnostics):
    # Write CSV with all columns + feedstock masses
    results_df.to_csv(f"crucible_ready_{date}.csv", index=False)
    
    # Generate PDF report with model card
    generate_pdf_report(model_diagnostics)
    
    # Add UI button
    export_btn = gr.Button("📥 Export for Synthesis")
    export_btn.click(export_for_lab, ...)

# Outcome logging
log_outcome_btn = gr.Button("✓ Log Outcome")
log_outcome_btn.click(log_outcome_in_csv, ...)
```

---

## Success Looks Like This

**Before (Current State):**

```
"I ran your model and got an alloy, but I have no idea:
- Whether it's chemically stable
- If my arc melter can actually make it
- How confident you are
- Whether to try this or skip it"
```

**After (Crucible-Ready):**

```
"I printed your CSV and walked to the furnace with:
- Clear stability score (0.92 = safe to try)
- My equipment matches the recommendation (arc melting ✓)
- Confidence levels tell me ML says 91%, rules say 92%, so 89% ensemble
- It's in-distribution, so I should trust the prediction
- Your estimate is $180 and 2.5 hours; I allocated exactly that
- If it works, I'll log it so your model learns from my studio"
```

---

## Timeline

| Phase | Weeks | What You Deliver |
|-------|-------|-----------------|
| **Data + Export** | 1 | Real MP data, valid compositions, lab-ready CSV |
| **Physics + UX** | 1 | Stability separation, process awareness, color codes |
| **Uncertainty** | 1 | Calibration, in-distribution, PDF report |
| **Testing** | 1 | Unit tests, end-to-end validation, one real synthesis |

---

## Three Documents to Keep Handy

1. **`crucible_roadmap.md`** (this repo)
   - Detailed implementation guide for each part
   - Code snippets and design decisions
   - ~600 lines of architectural guidance

2. **`implementation_checklist.md`** (this repo)
   - Checkbox-by-checkbox task list
   - Test cases for each feature
   - Progress tracking

3. **`studio_quick_ref.md`** (for the studio)
   - One-page user guide to print
   - Method reference table
   - Red flags & troubleshooting

---

## The Key Insight

**The tool is already good.** It has the right architecture, it can train and generate, it has an ML classifier and a rule-based system.

**What's missing is trust.**

Trust comes from:

- **Transparency**: "Here's exactly what I trained on and how I perform"
- **Uncertainty**: "I know when I'm extrapolating"
- **Actionability**: "The CSV tells you grams to weigh, not just a formula"
- **Feedback**: "Your outcome improves my next prediction"

Add those four things, and the studio will use it. Not because it's perfect, but because it's honest about what it is and how to use it.

---

## Next Action

**Today:**

1. Set `MP_API_KEY` environment variable (get from materialsproject.org)
2. Open `gradio_app.py`, find `creategradiointerface()`, locate `createsyntheticdataset()`
3. Replace with call to `gettrainingdataset()` from `materials_discovery_api.py`
4. Test: Run locally, verify 1000+ materials load, training completes

**By EOD:** Real MP data is feeding the model instead of fake data.

**By Day 3:** Stability and synthesizability are separate columns.

**By Day 5:** Studio can export and use the CSV without explanation.

**By Week 4:** The tool is crucible-ready.

---

Good luck! 🚀
