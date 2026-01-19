# START HERE: Crucible-Ready Materials Discovery Tool

Welcome. You have built something genuinely sophisticated: a VAE + ML classifier + rule engine pipeline for alloy discovery. Now you want to make it something a metallurgical studio will actually use.

This folder contains everything you need.

---

## 📋 Documents in This Folder

### 1. **ROADMAP_SUMMARY.md** ← START HERE
A 2-minute read that explains:
- What the problem is (you have a demo, not a tool)
- The solution in 3 bullets (real data, uncertainty, feedback)
- Timeline (4–5 weeks)
- Success criteria

**Read this first.**

---

### 2. **crucible_roadmap.md**
The detailed implementation guide (591 lines):
- Every change explained with rationale
- Code snippets for each major component
- Physics/chemistry reasoning behind thresholds
- All dependencies and integration points
- Optional enhancements

**Use this when building.**

---

### 3. **implementation_checklist.md**
Checkbox-by-checkbox task breakdown (598 lines):
- 40+ discrete tasks across 6 parts
- Test cases for each feature
- Progress tracking by week
- Success metrics
- Estimated hours per task

**Mark items off as you complete them.**

---

### 4. **studio_quick_ref.md**
One-page user guide (211 lines) for the metallurgists:
- How to use the tool (3 steps)
- What the numbers mean (table)
- Method reference (arc melting, solid state, furnace)
- Red flags and troubleshooting
- Example end-to-end run

**Print this for the workshop.**

---

## 🚀 Quick Start (Next 2 Hours)

### Hour 1: Set Up API
1. Go to [materialsproject.org](https://materialsproject.org)
2. Create account, request API key (free)
3. Set environment variable:
   ```bash
   export MP_API_KEY="your_key_here"
   ```
4. Test connection:
   ```python
   from materials_discovery_api import MaterialsProjectClient
   client = MaterialsProjectClient(apikey="your_key")
   test = client.getmaterialssummary(elements=["Al", "Ti"], limit=5)
   print(test)  # Should show 5 Al-Ti alloys
   ```

### Hour 2: Replace Mock Data
1. Open `gradio_app.py`
2. Find `creategradiointerface()` function
3. Locate line with `dataset = createsyntheticdataset(1000)`
4. Replace with:
   ```python
   from materials_discovery_api import gettrainingdataset
   mlfeatures, rawdata = gettrainingdataset(apikey=os.getenv("MP_API_KEY"), nmaterials=1000)
   ```
5. Run locally: `python gradio_app.py`
6. Verify: UI should show "Model trained on 1000 real binary alloys from Materials Project"

**Boom. Real data.**

---

## 📊 The 4-Week Sprint

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| **Week 1** | Real data + physics | Lab-ready CSV export |
| **Week 2** | Uncertainty + feedback | PDF report + outcome logging |
| **Week 3** | UX + testing | Model diagnostics dashboard |
| **Week 4** | Docs + validation | Hand off to studio |

---

## ✅ Success Criteria

When you can check all of these, you're done:

- [ ] Model is trained on **1000+ real Materials Project alloys** (not synthetic)
- [ ] All generated compositions are **chemically valid** (sum to 1.0, 5–95% bounds)
- [ ] **Stability and synthesizability are separate** columns with clear color codes
- [ ] **Process recommendations match actual equipment** (equipment selector in UI)
- [ ] **CSV export includes feedstock masses** (grams to weigh per 100g batch)
- [ ] **Uncertainty is explicit** (calibration flag, in-distribution status, confidence scores)
- [ ] **Outcome logging works** (studio can log successes/failures)
- [ ] **PDF report shows model metrics** (accuracy, precision, recall, F1, CV scores)
- [ ] **All unit tests pass** (data loading, composition validity, ML performance)
- [ ] **One real alloy synthesized** and outcome logged successfully
- [ ] **Studio technician can print CSV and use it** without asking questions

---

## 💡 Key Insight

**You already have a good tool.** The architecture is right, the pipeline works, the models are trained.

What's missing is **trust**. And trust comes from:

1. **Real data** (not random numbers)
2. **Clear uncertainty** (not just 0–1 probability)
3. **Actionable outputs** (feedstock masses, not formulas)
4. **Closed feedback loop** (outcomes improve the model)

Add those four things in 4 weeks, and the studio will use it.

---

## 📞 When You Get Stuck

### If API errors:
- Check MP_API_KEY is set: `echo $MP_API_KEY`
- Test connection in Python REPL first (before Gradio)
- MP API rate-limits at ~5 req/sec; code already has sleeps

### If composition constraints fail:
- Check `generatematerials()` in gradio_app.py
- Ensure: `comp1 = np.clip(features[0], 0.05, 0.95)`
- Then: `comp2 = 1.0 - comp1`
- Assert: `assert abs(comp1 + comp2 - 1.0) < 1e-6`

### If UI doesn't show real data:
- Verify `mlfeatures` has > 0 rows
- Check `mlclassifier.train()` completes
- Look for error messages in terminal

### If tests fail:
- Run one test at a time: `pytest test_synthesizability.py::test_name -v`
- Check test data assumptions (e.g., does test MP query work?)
- Add `print()` statements to debug

---

## 📖 Document Map

```
START_HERE.md (you are here)
├── ROADMAP_SUMMARY.md (read first, 250 lines)
├── crucible_roadmap.md (reference while building, 591 lines)
├── implementation_checklist.md (track progress, 598 lines)
└── studio_quick_ref.md (print for workshop, 211 lines)
```

---

## 🎯 Next Actions

1. **Now**: Read `ROADMAP_SUMMARY.md` (10 minutes)
2. **Today**: Complete "Quick Start" above (2 hours)
3. **Tomorrow**: Open `implementation_checklist.md`, start Part 1
4. **Next 4 weeks**: Follow the roadmap, checking off items

---

## The Studio Will Love This When:

They can:
- Generate 50 alloy candidates
- Print a CSV
- Walk to the furnace with exact temperatures, atmospheres, and feedstock masses
- Try the top 5, log which ones succeed
- See their data reflected back in hit rates
- Know the model is learning from their real metallurgy

That's not a demo anymore. That's a lab resource.

---

**Go build it. 🚀**

Questions? See `crucible_roadmap.md` or `studio_quick_ref.md`.