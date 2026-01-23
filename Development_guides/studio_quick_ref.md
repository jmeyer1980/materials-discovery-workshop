# Materials Discovery Tool: Quick Reference for the Workshop

## What This Tool Does

Generates novel alloy compositions and predicts whether they can be synthesized with your equipment. Designed for metallurgical studios to explore new materials with confidence.

---

## Before You Start: What You Need

- **Materials Project API Key** (free from materialsproject.org)
- **Your equipment list**: Arc melter? Furnace (max temp)? Vacuum chamber? Rolling mill?
- **Starting capital budget**: e.g., $500/experiment, $5000/series

---

## The Workflow in 3 Steps

### Step 1: Generate & Predict

1. Click **"Generate Materials"** in the app
2. Adjust sliders:
   - **Latent Dimension** (5–20): Higher = more variety in generated alloys
   - **Training Epochs** (10–200): Higher = better learned patterns (slower)
   - **Materials to Generate** (10–500): How many candidates you want
3. Wait 30–60 seconds
4. See results: Formula, stability, synthesizability probability, cost estimate

### Step 2: Review Results

Look for alloys with:

- ✅ **Green row**: High stability (Ehull < 0.025) + high synth probability (> 0.8) = **PRIORITY 1**
- ⚠️  **Yellow row**: Stable (Ehull < 0.1) but uncertain synth = **PRIORITY 2, validate first**
- ❌ **Red row**: Unstable (Ehull > 0.1) = **Risky, skip unless exploring**

**Key columns:**

- **Stability**: Color-coded; 0.95 = chemically sound, 0.5 = iffy
- **Synth Prob**: ML + expert rules combined; > 0.85 = high confidence
- **Confidence**: Higher = model has seen similar alloys before
- **In Distribution**: Green = alloy is similar to known materials; Red = novel, use as inspiration only
- **Recommended Method**: Arc melt / Furnace / CVD based on your equipment
- **Est. Cost**: Material + equipment + labor for one attempt
- **Success Prob**: Chance it works given the method and material

### Step 3: Execute & Log

1. Click **"Export for Synthesis"** → Download CSV + PDF
2. Print the top 5–10 candidates
3. In the lab:
   - Weigh out ingredients (CSV has grams per 100g batch)
   - Follow recommended method and temperature
   - Log result: Success / Partial / Failed
4. Back in the tool:
   - Log outcome in "Experimental Outcomes" section
   - Model improves over time with your real results

---

## Understanding the Numbers

| Column | Meaning | Green | Yellow | Red |
|--------|---------|-------|--------|-----|
| **Ehull** (eV/atom) | Thermodynamic stability | < 0.025 | 0.025–0.1 | > 0.1 |
| **Stability Score** | Composite stability | > 0.8 | 0.5–0.8 | < 0.5 |
| **Synth Prob** | Predicted probability of synthesis | > 0.8 | 0.5–0.8 | < 0.5 |
| **ML Confidence** | How sure the ML model is | > 0.85 | 0.7–0.85 | < 0.7 |
| **Priority Rank** | 1 = try first; 10 = last | 1–3 | 4–7 | 8–10 |
| **In Distribution** | Similar to training data | In | — | Out |

---

## Recommended Methods & Equipment

### Arc Melting

- **Best for:** Low-melting-point alloys, small batches
- **Temp range:** 1200–3000°C
- **Atmosphere:** Argon or vacuum
- **Prep:** 2 hours
- **Synthesis:** 0.5 hours
- **Cost:** ~$150/run (equipment + materials + labor)
- **Success base:** 80%

### Solid State Reaction

- **Best for:** Stable, low-melting alloys; oxides
- **Temp range:** 600–1200°C
- **Atmosphere:** Air or Argon
- **Prep:** 4 hours (powder mixing, grinding)
- **Synthesis:** 4–8 hours
- **Cost:** ~$100/run
- **Success base:** 70%

### Furnace (1000–1500°C)

- **Best for:** Most binary alloys
- **Temp:** Depends on your furnace; ensure > melting points of both elements
- **Atmosphere:** Inert (Argon) preferred
- **Prep:** 2 hours
- **Synthesis:** 2–4 hours
- **Cost:** ~$120/run
- **Success base:** 75%

---

## Red Flags: When NOT to Synthesize

| Flag | Reason |
|------|--------|
| **Ehull > 0.15** | Thermodynamically unstable; very unlikely to form |
| **Synth Prob < 0.4** | Model predicts low success; wait for better candidates |
| **Out of Distribution** | Novel; more exploratory than predictive |
| **Overconfident flag** | Model is making uncertain predictions; cross-check with literature |
| **High n_sites (> 20)** | Large, complex unit cell; harder to synthesize experimentally |
| **Extreme density (< 2 or > 30)** | Unusual; may indicate calculation error |

---

## Interpreting Confidence Intervals

After you log 10+ outcomes, the tool will show you calibration metrics:

- **Well-calibrated**: "If the model says 80%, ~80% of those actually succeed" → Trust probabilities
- **Overconfident**: "Model says 80%, but only 60% succeed" → Be more skeptical of high-prob predictions
- **Underconfident**: "Model says 40%, but 60% succeed" → More optimistic possibilities

---

## Feedback Loop: How Your Results Improve the Tool

1. **You synthesize an alloy**: Success or fail
2. **You log it in the tool**: Formula, outcome, notes
3. **System tracks**: Predicted prob vs. actual outcome
4. **Monthly update**: Model retrains on your outcomes
5. **Next batch**: Predictions are tuned to **your** equipment, your materials, your skill level

This is how the tool becomes yours; it learns from your metallurgical expertise.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "All generated alloys are red (unstable)" | Lower the "risk tolerance" slider (strict stability filter). Or increase latent dimension to explore more. |
| "I don't trust the probabilities" | Check "Model Diagnostics" tab for accuracy/precision/recall. Compare with literature for test alloys. |
| "Recommended method isn't available" | Go to "Equipment Setup" and tell the tool what you have. Recommendations will filter automatically. |
| "I synthesized this and it failed, but model said 90% success" | Log it! The model will learn. Also check "Safety & Limitations"—you may be in an extrapolation region. |
| "CSV export is missing columns" | Ensure you have the latest app version. Contact support with your setup. |

---

## Safety & Disclaimers

⚠️ **This tool makes predictions, not guarantees.**

- Predictions are based on **thermodynamics**, not kinetics or experimental realities.
- Model is trained on computational data (Materials Project), not lab-tested alloys.
- Equipment, processing history, impurities, and operator skill all affect outcomes.
- **Always** validate with small-scale tests before scaling.
- **Never** assume 80% success probability = you'll definitely succeed.
- Feedback from your experiments refines the model over time; early predictions are less reliable.

---

## Quick Start Checklist

- [ ] Set API key (environment variable `MP_API_KEY`)
- [ ] Tell the tool your equipment (equipment selector)
- [ ] Generate 50 alloys (default settings fine)
- [ ] Review top 5 candidates (green/yellow/red filter)
- [ ] Try the highest-priority one (smallest batch first)
- [ ] Log result in tool
- [ ] After 5–10 trials, compare with tool predictions
- [ ] Give feedback so model learns

---

## Contact & Resources

- **Materials Project**: materialsproject.org
- **Pymatgen Docs**: pymatgen.org/docs
- **ICSD Database**: inorganic-chemistry.org
- **Literature Search**: Google Scholar, ResearchGate for alloy synthesis papers

---

## Example: From Tool to Crucible

**Tool output:**

```
Formula: Al0.6Ti0.4
Stability: 0.92 (Ehull = 0.018 eV/atom)
Synth Prob: 0.87 (ML: 0.91, LLM: 0.81)
Method: Arc Melting
Feedstock (100g): 59.3g Al, 40.7g Ti
Est. Cost: $180
Est. Time: 2.5 hours
Success Prob: 84%
Priority: 1 (TRY THIS FIRST)
```

**You do:**

1. Print and bring to shop
2. Weigh 59.3g Al, 40.7g Ti
3. Arc melt in Ar at 1800°C for 20 min
4. Cool to room temp
5. Characterize (SEM, XRD, hardness)
6. Log result: "Success! Clean ingot, no cracks. Hardness 520 HV."
7. Back in tool: Click "Success" and add notes
8. Next week: Model re-trains; your feedback improves predictions for you and others

---

**You're now running the tool like a lab resource, not a black box.**

Good luck! 🔬⚙️
