# Materials Discovery Workshop: Roadmap towards Crucible Experiment

## Detailed Implementation Guide

This document is your detailed playbook. Read the ROADMAP_SUMMARY.md first. Then use this as your reference while building.

---

## Part 1: Data & Training (Week 1)

### 1.1 Replace Mock Synthesizability Labels with Real MP Signals

**Current state:**

- `synthesizability_predictor.py` uses mock ICSD vs MP-only distributions.
- No real "did people synthesize this?" ground truth.

**What to do:**

1. In `synthesizability_predictor.py`:
   - Replace `generatemockicsddata()` with a real ICSD lookup or proxy:
     - Use Materials Project's `is_stable` field (available via API).
     - Flag materials with `energy_above_hull < 0.025 eV/atom` as "experimentally likely" (high success rate in literature).
     - Flag materials with `energy_above_hull > 0.1 eV/atom` as "risky" (rarely synthesized).
   - Replace `generatemockmponlydata()` with MP materials filtered as above, split by stability.

2. Call `gettrainingdataset()` from `materials_discovery_api.py` at startup instead of mocking:

   ```python
   # In gradio_app.py init
   mlclassifier = SynthesizabilityClassifier()
   mlfeatures, rawdata = gettrainingdataset(apikey=os.getenv("MP_API_KEY"), nmaterials=1000)
   mlmetrics = mlclassifier.train_on_real_data(mlfeatures)
   ```

3. Externalize the API key:
   - Move the hardcoded key to `environment` variable `MP_API_KEY`.
   - Add fallback: if key missing, warn user and use synthetic data with a clear banner.

**Why this works:**

- The classifier now learns from actual MP chemical space instead of random distributions.
- Stability thresholds (0.025, 0.1 eV/atom) are standard in the literature—materials folks will recognize them.
- Real vs. synthetic mode is transparent in the UI.

**Code changes (estimate 30 mins):**

- Edit `synthesizability_predictor.py`: Replace mock functions.
- Edit `gradio_app.py`: Add API key handling, call real dataset at startup.

---

### 1.2 Train VAE on Real MP Feature Space

**Current state:**

- VAE trained on synthetic compositions (Al, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn random).
- Generated candidates may not reflect real alloy chemistry.

**What to do:**

1. In `gradio_app.py`, after fetching real MP data:
   - Use `mlfeatures` DataFrame columns directly for VAE training instead of `createsyntheticdataset()`.
   - Log the data source in the UI: "Model trained on **500 real binary alloys** from Materials Project."

2. Keep compositions bounded and valid:

   ```python
   # In generatematerials()
   comp1 = np.clip(features[0], 0.05, 0.95)  # 5–95% bounds
   comp2 = 1.0 - comp1
   assert abs(comp1 + comp2 - 1.0) < 1e-6
   ```

**Why this works:**

- Generated alloys live in the same chemical space as known materials.
- Compositions are guaranteed chemically valid (sum to 1.0, within bounds).
- Someone in the shop sees "trained on 500 real alloys" and feels confident.

**Code changes (estimate 20 mins):**

- Edit `generatematerials()` to enforce composition bounds.
- Log dataset source in UI markdown.

---

## Part 2: Physics & Chemistry Rigor (Week 1–2)

### 2.1 Standardize Stability Thresholds

**Current state:**

- Rule-based predictor uses hardcoded thresholds (e.g., energy above hull < 0.1).
- Not documented or adjustable.

**What to do:**

1. In `synthesizability_predictor.py`, create a config dict:

   ```python
   STABILITY_THRESHOLDS = {
       "highly_stable": {"energy_above_hull_max": 0.025, "description": "Likely synthesized"},
       "marginal": {"energy_above_hull_max": 0.1, "description": "Possible with care"},
       "risky": {"energy_above_hull_max": float('inf'), "description": "Unstable, unlikely"}
   }
   ```

2. In `gradio_app.py`, expose a slider:

   ```python
   risk_tolerance = gr.Slider(
       minimum=0.025, maximum=0.2, value=0.1, step=0.025,
       label="Energy Above Hull Threshold (eV/atom)",
       info="Lower = stricter filter for stable materials"
   )
   ```

3. Pass `risk_tolerance` to the classifier at prediction time.

**Why this works:**

- Metallurgists understand "I'm willing to accept materials up to 0.1 eV/atom above hull."
- Thresholds match literature conventions.
- Studio can adjust based on their risk appetite and equipment.

**Code changes (estimate 1 hour):**

- Add config to `synthesizability_predictor.py`.
- Add UI slider to `gradio_app.py`.
- Thread `risk_tolerance` through predict pipeline.

---

### 2.2 Separate Stability from Synthesizability

**Current state:**

- `LLMSynthesizabilityPredictor` mixes thermodynamic stability (energy above hull) with synthesizability (can you make it?).

**What to do:**

1. Create two separate columns:
   - **Thermodynamic Stability Score** (0–1): Based on Ehull, formation energy, density, etc. Color code: red (unstable) → yellow (marginal) → green (stable).
   - **Synthesizability Probability** (0–1): Based on ML classifier + rule-based heuristics (complexity, element compatibility, etc.).

2. In `gradio_app.py`, show both in the results table:

   ```
   | Formula  | Stability | E_hull | Synth Prob | ML Conf | LLM Conf | Recommended |
   |----------|-----------|--------|-----------|---------|----------|-------------|
   | Al0.5Ti  | 0.95      | 0.02   | 0.87      | 0.92    | 0.81     | ✓ Try this  |
   ```

3. Add explicit UI notes:
   - Green row: "Stable AND likely synthesizable → HIGH PRIORITY"
   - Yellow row: "Stable but unclear synthesizability → MEDIUM PRIORITY"
   - Red row: "Unstable → Risky, not recommended"

**Why this works:**

- Metallurgists can see at a glance which materials are chemically sound vs. experimentally feasible.
- Decoupling allows them to pursue "unstable" materials if they're curious (marked as risky).
- Transparent prioritization builds confidence.

**Code changes (estimate 1.5 hours):**

- Edit `synthesizability_predictor.py`: Add stability scoring function.
- Edit `gradio_app.py`: Add two-column display, color coding.

---

### 2.3 Process-Aware Recommendations

**Current state:**

- `costbenefitanalysis()` assigns methods based on bandgap and stability (generic).
- Precursor recommendations hardcoded.

**What to do:**

1. Add equipment and atmosphere tagging:

   ```python
   SYNTHESIS_METHODS = {
       "arc_melting": {
           "temp_range": (1200, 3000),  # derived from melting points
           "atmosphere": "Argon or vacuum",
           "equipment_needed": ["arc melter", "graphite crucible"],
           "prep_time_hours": 2,
           "synthesis_time_hours": 0.5,
           "success_probability_base": 0.8,
       },
       "solid_state_reaction": {
           "temp_range": (600, 1200),
           "atmosphere": "Air or Argon",
           "equipment_needed": ["furnace", "ceramic crucible"],
           "prep_time_hours": 4,
           "synthesis_time_hours": 4,
           "success_probability_base": 0.7,
       },
   }
   ```

2. In `gradio_app.py`, add equipment selector:

   ```python
   available_equipment = gr.CheckboxGroup(
       choices=["Arc Melter", "Furnace (1200°C)", "Vacuum Chamber", "Rolling Mill"],
       label="Available Equipment",
       info="Filter recommendations to your setup"
   )
   ```

3. Filter `costbenefitanalysis()` results to match available equipment.

**Why this works:**

- "Use arc melting" is generic; "Use arc melting, 45 min at 2200°C under Ar, costs ~$150 in materials" is actionable.
- Studio sees immediately if a candidate is feasible with their gear.
- Removes guesswork about what's possible in-house.

**Code changes (estimate 1.5 hours):**

- Add SYNTHESIS_METHODS config.
- Add equipment selector to UI.
- Filter recommendations in costbenefitanalysis().

---

### 2.4 Tight Composition Constraints

**Current state:**

- Generated compositions may be invalid (> 100%, < 0%, etc.).
- Single element per slot; no exploration of, e.g., (Al0.5Ti0.3V0.2).

**What to do:**

1. For binary alloys (primary use case):

   ```python
   # In generatematerials()
   comp1 = np.clip(features[0], 0.05, 0.95)
   comp2 = 1.0 - comp1
   assert sum([comp1, comp2]) == 1.0
   ```

2. For ternary alloys (advanced):

   ```python
   comp1 = np.clip(features[0], 0.05, 0.85)
   comp2 = np.clip(features[1], 0.05, 0.85)
   comp3 = 1.0 - comp1 - comp2
   if comp3 < 0.05:  # Reallocate if too small
       comp1, comp2, comp3 = 0.5, 0.3, 0.2  # Fallback
   assert abs(sum([comp1, comp2, comp3]) - 1.0) < 1e-6
   ```

3. Add export formats:

   ```python
   # Export CSV includes at.%, wt.%, and feedstock mass for 100g batch
   export_df["%_atm"] = comp1 * 100
   export_df["%_wt"] = calc_wt_percent(elements, compositions)
   export_df["feedstock_g_per_100g"] = calc_feedstock(...)
   ```

**Why this works:**

- Studio can print the CSV and immediately weigh out ingredients for a 100g batch.
- "Al0.5Ti0.5" → "51g Al, 49g Ti" is in the same language they use.
- No ambiguity about what to put in the crucible.

**Code changes (estimate 1 hour):**

- Tighten composition generation in `generatematerials()`.
- Add weight-percent calculations (use `pymatgen.core.Composition`).
- Add feedstock columns to export CSV.

---

## Part 3: Uncertainty & Validation (Week 2)

### 3.1 Calibration & In-Distribution Signals

**Current state:**

- ML classifier outputs 0–1 probability with no indication of reliability.
- No signal for "this is outside the training set."

**What to do:**

1. Compute calibration metrics at startup:

   ```python
   # In SynthesizabilityClassifier.train()
   from sklearn.calibration import calibration_curve
   prob_true, prob_pred = calibration_curve(ytest, ypred_proba, n_bins=10)
   # Store and plot in UI
   ```

2. Add "calibration" column to results:

   ```python
   # For each predicted probability, compute distance to calibration curve
   # If model is well-calibrated: high probability ≈ high actual frequency
   # If overconfident: high probability >> actual frequency
   results_df["calibration_flag"] = "well_calibrated" | "overconfident"
   ```

3. Add in-distribution detection:

   ```python
   # Compute distance in latent space or feature space
   # Materials near training data: "in-distribution"
   # Materials far away: "extrapolation – treat as exploratory"
   results_df["distribution_status"] = "in_dist" | "out_dist"
   results_df["nn_distance"] = nearest_neighbor_distance_to_training
   ```

**Why this works:**

- Studio sees: "This is a high-probability prediction with high confidence" vs. "This is a novel extrapolation; use as inspiration, not gospel."
- Builds trust by being explicit about limits.
- Matches how domain experts think: "Is this thing I'm predicting like things I've seen before?"

**Code changes (estimate 2 hours):**

- Add calibration computation to `SynthesizabilityClassifier.train()`.
- Add KNN distance calculation to generation pipeline.
- Add columns to results table.
- Add visualization tab (calibration curve, distance histogram).

---

### 3.2 Export with Full Context

**Current state:**

- UI shows results, but no easy way to export for the lab.

**What to do:**

1. Create a single "Export for Synthesis" button that generates:

   ```
   crucible_ready_alloys_20260118.csv
   ```

   With columns:

   ```
   | ID | Formula | Comp1 | Comp2 | at% | wt% | Feed_g_100g | Stability | Synth_Prob | ML_Conf | LLM_Conf | Ensemble_Conf | Ehull | Method | Precursors | Temp_C | Atm | Success_Prob | Est_Cost | Est_Time_h | Priority | In_Distribution | NN_Distance | Notes |
   |----|---------| ... |
   ```

2. Add a companion PDF with:
   - Model summary (accuracy, precision, recall, F1, CV scores).
   - Data source: "1000 materials from Materials Project, filtered by energy above hull < 0.1 eV/atom".
   - Limitations: "Model trained on binary alloys. Ternary predictions are extrapolation."
   - Recommended workflow: "Start with Priority 1, highest confidence candidates."

3. In `gradio_app.py`:

   ```python
   export_btn = gr.Button("Export Synthesis Plan")
   export_btn.click(
       export_for_lab,
       inputs=[results_df, model_diagnostics],
       outputs=[gr.File(label="Download CSV"), gr.File(label="Download Report PDF")]
   )
   ```

**Why this works:**

- Metallurgist clicks one button, walks away with a printed CSV and a checklist.
- Full context (stability, confidence, success odds, cost, time) is right there.
- No ambiguity—the export is a concrete contract between the model and the experimentalist.

**Code changes (estimate 2–3 hours):**

- Create `export_for_lab()` function in `gradio_app.py`.
- Generate PDF with `reportlab` or similar (lightweight).
- Ensure all data is logged (costs, times, success probs) and surfaced.

---

### 3.3 Feedback Loop Setup

**Current state:**

- Model is static; no way to learn from failed or successful experiments.

**What to do:**

1. Add "Experimental Outcome Log" in the UI:

   ```python
   outcome = gr.Dropdown(
       choices=["Not yet attempted", "Successful synthesis", "Partial success", "Failed"],
       label="Outcome",
       interactive=True
   )
   notes = gr.Textbox(label="Notes (optional)")
   add_outcome_btn = gr.Button("Log Outcome")
   ```

2. Store outcomes in a simple CSV or JSON:

   ```
   outcomes.csv:
   formula, attempted_date, outcome, ml_pred, llm_pred, ehull, notes
   Al0.5Ti0.5, 2026-01-20, Successful synthesis, 0.87, 0.92, 0.02, "Melted cleanly at 1800°C"
   ```

3. Add "Validation metrics" tab:
   - Precision/recall on outcomes to date.
   - "In last 10 experiments: 7 successes (70% hit rate) among high-confidence predictions."

**Why this works:**

- Model improves over time as the studio feeds back real results.
- Studio sees their own data reflected back: "Our metallurgy validates the model."
- Builds long-term trust and continuous improvement loop.

**Code changes (estimate 2 hours):**

- Add outcome logging UI to `gradio_app.py`.
- Add CSV writer to save outcomes.
- Add validation metrics computation.
- Optionally, periodically retrain on real outcomes (weekly batch job).

---

## Part 4: UX Polish (Week 3)

### 4.1 Model Diagnostics Dashboard

**Add a dedicated tab:**

```
| Model Diagnostics Tab |
|   Training Data:
|     - Source: Materials Project
|     - Count: 1,000 materials
|     - Element systems: Al-Ti, Al-Ni, Al-Cu, Fe-Co, etc.
|     - Date: 2026-01-18
|
|   ML Classifier (Random Forest):
|     - Accuracy: 87%
|     - Precision: 0.91
|     - Recall: 0.83
|     - F1: 0.87
|     - CV Mean: 0.85 ± 0.04
|
|   Calibration:
|     - [Plot: Predicted vs. Observed Frequency]
|     - Status: Well-calibrated
|     - Recommendation: Trust probabilities as-is
|
|   Coverage:
|     - In-distribution: 78% of generated candidates
|     - Extrapolation: 22% (novel compositions)
|
|   Last Updated: 2026-01-18 12:34 UTC
```

**Code changes (estimate 1.5 hours):**

- Create metrics computation function.
- Add Markdown + Plot to Gradio.

---

### 4.2 Safety & Limitations Panel

**Add a visible note:**

```
⚠️ LIMITATIONS & SAFETY
- Predictions are statistical, not deterministic.
- Model trained on Materials Project calculations; real synthesis may differ.
- For novel element combinations, treat as exploratory.
- Always validate with literature and lab-scale experiments before scaling.
- See "Model Diagnostics" for full performance metrics.
```

**Code changes (estimate 20 mins):**

- Add Markdown block in `gradio_app.py`.

---

## Part 5: Validation & Testing (Week 3–4)

### 5.1 Unit & Integration Tests

**In `test_synthesizability.py`, add:**

- ✓ Real MP data can be loaded and features extracted.
- ✓ VAE trains on real data and generates valid compositions (sum ≈ 1.0).
- ✓ ML classifier has reasonable metrics on held-out test set.
- ✓ LLM predictor rules are consistent with thresholds.
- ✓ CSV export has all required columns.
- ✓ Equipment filtering works correctly.
- ✓ In-distribution detection flags outliers.

**Code changes (estimate 2 hours):**

- Expand `test_synthesizability.py`.
- Add integration test with small real MP dataset.

---

### 5.2 End-to-End Demo

**Run a full pipeline on real data:**

1. Load 100 real binary alloys from MP.
2. Train VAE + ML classifier.
3. Generate 50 new candidates.
4. Predict synthesizability.
5. Export CSV.
6. Manually inspect: Do the high-priority candidates look chemically reasonable?

**Expected output:**

- Top 5 candidates are mixtures of real binary alloys or slight variations.
- Stability scores correlate with Ehull.
- Success probabilities make intuitive sense (high for low-Ehull, low for high-Ehull).

**Code changes (estimate 1 hour):**

- Run notebook end-to-end with real data.
- Sanity-check outputs.

---

## Part 6: Metadata & Documentation (Week 4)

### 6.1 Model Card

Create a one-page "model card" (inspired by Google/Hugging Face):

```
# Materials Discovery VAE + Synthesizability Classifier

## Intended Use
Generate novel binary alloy candidates and predict synthesizability for experimental validation.

## Training Data
- Source: Materials Project API
- Count: 1,000 binary alloys
- Stability filter: energy_above_hull < 0.1 eV/atom
- Features: composition, density, formation energy, band gap, electronegativity, atomic radius, n_sites

## Model Architecture
- VAE: 6→64→32→[5-dim latent]→32→64→6
- Classifier: Random Forest, 200 estimators, max_depth=10
- Ensemble: 70% RF + 30% rule-based heuristics

## Performance
- Accuracy: 87%
- Precision: 0.91 (few false positives)
- Recall: 0.83 (catches most synthesizable materials)
- F1: 0.87
- 5-fold CV: 0.85 ± 0.04

## Limitations
- Binary alloys only (extension to ternary in progress).
- Does not account for kinetic barriers (only thermodynamics).
- May overestimate success for novel element combinations.
- No account for impurities, grain size, or processing history.

## Recommended Workflow
1. Start with high-confidence, high-priority candidates.
2. Validate predictions experimentally at lab scale.
3. Log outcomes to refine model over time.
```

---

## Timeline & Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1    | Data + VAE | Real MP data, valid compositions, trained VAE |
| 1–2  | Physics   | Stability thresholds, process awareness, composition constraints |
| 2    | Export    | Lab-ready CSV with full context |
| 2–3  | UX        | Diagnostics, limitations, feedback UI |
| 3–4  | Testing   | Unit tests, integration tests, end-to-end validation |
| 4    | Docs      | Model card, API docs, user guide |

---

## Success Criteria

When you can answer "yes" to all of these, you're crucible-ready:

1. **Data realism:** "The model is trained on 1000+ real alloys from Materials Project with a clear stability criterion."
2. **Physics transparency:** "I can see why a material is recommended: thermodynamic stability, synthesizability probability, equipment fit, cost/benefit."
3. **Composition validity:** "The CSV tells me exactly how many grams of each element to weigh out for a 100g batch."
4. **Uncertainty bounds:** "High-confidence vs. exploratory candidates are clearly flagged. I know when I'm extrapolating."
5. **Export precision:** "I can print the CSV and walk into the shop with a concrete plan."
6. **Feedback closure:** "My outcomes (success/failure) are logged and influence future predictions."
7. **Model transparency:** "I see the accuracy, precision, recall, and cross-validation metrics. I know the model's limits."
8. **Equipment match:** "The recommendations fit my actual gear (arc melter, furnace, etc.)."

---

## Optional Enhancements (Beyond MVP)

- **Property prediction:** Predict melting point, Young's modulus, electrical conductivity based on generated composition.
- **Ternary alloys:** Extend VAE and classifier to handle ternary (Al-Ti-Ni, etc.).
- **Reinforcement learning:** Optimize for specific properties (e.g., "generate high-strength alloys").
- **Literature mining:** Auto-fetch related papers for top candidates.
- **Kinetic assessment:** Layer in activation energy / diffusion barriers.

---

Good luck with implementation! Refer back to this document as your detailed reference guide.
