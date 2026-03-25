#!/usr/bin/env python3
"""
Materials Discovery Workshop — CLI
====================================
Exposes the core API surface as safe, scriptable commands for debugging,
validation, and artifact management.

Usage:
    python cli.py <command> [options]
    python cli.py --help

Commands:
    validate-key      Validate a Materials Project API key.
    fetch-materials   Fetch materials data from the Materials Project API
                      and save to a local CSV.
    check-safety      Run the lab-safety filter against a local CSV file and
                      print a pass/fail summary.
    export-lab        Generate the full safe-ish lab export (CSV + PDF) from a
                      local predictions CSV and save artifacts to a directory.
    export-csv        Generate the generic all-materials analysis CSV from a
                      local predictions CSV.
    status            Show environment status: API key presence, hazard config,
                      field mapping, and ML feature schema.

Environment variables:
    MP_API_KEY        Materials Project API key (used by validate-key and
                      fetch-materials when --api-key is not supplied).

Security note:
    API keys are never printed in full — the CLI always masks all but the
    first 4 characters.
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """Return a masked version of an API key for display (first 4 + stars)."""
    if not key:
        return "(none)"
    return key[:4] + "*" * (len(key) - 4) if len(key) > 4 else "****"


def _resolve_output_dir(raw: str) -> Path:
    """
    Return an absolute, validated output directory path.
    Creates the directory if it does not exist.
    Raises SystemExit if the resolved path escapes a reasonable boundary.
    """
    path = Path(raw).resolve()
    # Refuse paths that look like obvious traversals (starts with /etc, /sys …)
    blocked = ["/etc", "/sys", "/proc", "/dev", "C:\\Windows\\System32"]
    for b in blocked:
        if str(path).startswith(b):
            sys.exit(f"[ERROR] Output directory '{path}' is not allowed.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_predictions_csv(csv_path: str):
    """Load a predictions CSV and return a pandas DataFrame. Exits on error."""
    import pandas as pd
    p = Path(csv_path).resolve()
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {p}")
    if p.suffix.lower() != ".csv":
        sys.exit(f"[ERROR] Expected a .csv file, got: {p.suffix}")
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot read CSV '{p}': {exc}")
    if df.empty:
        sys.exit(f"[ERROR] CSV file is empty: {p}")
    return df


def _normalize_for_safety_checks(df):
    """Normalize known lab-export column names to internal safety-check schema."""
    normalized = df.copy()

    column_aliases = {
        'SynthProb': 'ensemble_probability',
        'Stability': 'thermodynamic_stability_category',
        'InDist': 'in_distribution',
        'Formula': 'formula',
        'NNDistance': 'nn_distance',
    }

    for source, target in column_aliases.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]

    if 'ensemble_probability' in normalized.columns:
        normalized['ensemble_probability'] = normalized['ensemble_probability'].astype(str).str.replace('%', '', regex=False)
    if 'in_distribution' in normalized.columns:
        normalized['in_distribution'] = normalized['in_distribution'].astype(str).str.strip().str.lower()

    return normalized


def _print_section(title: str, width: int = 60) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Command: validate-key
# ---------------------------------------------------------------------------

def cmd_validate_key(args: argparse.Namespace) -> None:
    """Validate a Materials Project API key."""
    _print_section("Validate API Key")

    api_key = args.api_key or os.getenv("MP_API_KEY")
    if not api_key:
        print("[WARN] No API key supplied via --api-key or MP_API_KEY env var.")
        print("       Get a free key at https://materialsproject.org/api")
        sys.exit(1)

    print(f"  Key: {_mask_key(api_key)}")
    print("  Testing connection to Materials Project …")

    try:
        from api_authentication_handler import APIAuthenticationHandler
        handler = APIAuthenticationHandler(api_key=api_key)
        result = handler.validate_api_key()
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import APIAuthenticationHandler: {exc}")

    if result.get("valid"):
        print(f"  [OK] {result.get('message', 'Key is valid.')}")
    else:
        print(f"  [FAIL] {result.get('message', 'Validation failed.')}")
        for suggestion in result.get("suggestions", []):
            print(f"    • {suggestion}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command: fetch-materials
# ---------------------------------------------------------------------------

def cmd_fetch_materials(args: argparse.Namespace) -> None:
    """Fetch materials from the Materials Project API and save to a CSV."""
    _print_section("Fetch Materials from Materials Project")

    api_key = args.api_key or os.getenv("MP_API_KEY")
    if not api_key:
        print("[WARN] No API key. Using synthetic/fallback dataset instead.")

    out_dir = _resolve_output_dir(args.output_dir)

    elements = [e.strip() for e in args.elements.split(",") if e.strip()] if args.elements else None
    limit = args.limit
    hull_max = args.hull_max

    print(f"  Elements filter : {elements or '(none — all binary alloys)'}")
    print(f"  Energy hull max : {hull_max} eV/atom")
    print(f"  Row limit       : {limit}")
    print(f"  Output dir      : {out_dir}")

    try:
        from materials_discovery_api import MaterialsProjectClient, create_ml_features_from_mp_data
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import materials_discovery_api: {exc}")

    if not api_key:
        # Fall back to synthesizability predictor's synthetic dataset
        try:
            from synthesizability_predictor import create_synthetic_dataset_fallback
            df = create_synthetic_dataset_fallback(n_samples=limit)
            filename = f"synthetic_fallback_{limit}.csv"
            outpath = out_dir / filename
            df.to_csv(outpath, index=False)
            print(f"  [OK] Saved synthetic fallback dataset ({len(df)} rows) → {outpath}")
        except Exception as exc:
            sys.exit(f"[ERROR] Fallback dataset generation failed: {exc}")
        return

    try:
        client = MaterialsProjectClient(api_key=api_key)

        if elements:
            raw_df = client.get_materials_summary(
                elements=elements,
                energy_above_hull_max=hull_max,
                limit=limit,
            )
        else:
            raw_df = client.get_binary_alloys(
                limit_per_pair=max(10, limit // 20),
                energy_above_hull_max=hull_max,
            )
    except Exception as exc:
        sys.exit(f"[ERROR] API fetch failed: {exc}")

    if raw_df.empty:
        print("  [WARN] No materials returned for the given filters.")
        sys.exit(1)

    print(f"  Fetched {len(raw_df)} raw records. Building ML features …")

    try:
        ml_df = create_ml_features_from_mp_data(raw_df)
    except Exception as exc:
        print(f"  [WARN] ML feature generation failed ({exc}). Saving raw data.")
        ml_df = raw_df

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = out_dir / f"mp_raw_{ts}.csv"
    ml_file = out_dir / f"mp_ml_features_{ts}.csv"

    raw_df.to_csv(raw_file, index=False)
    ml_df.to_csv(ml_file, index=False)

    print(f"  [OK] Raw data    ({len(raw_df)} rows)        → {raw_file}")
    print(f"  [OK] ML features ({len(ml_df)} rows × {len(ml_df.columns)} cols) → {ml_file}")


# ---------------------------------------------------------------------------
# Command: check-safety
# ---------------------------------------------------------------------------

def cmd_check_safety(args: argparse.Namespace) -> None:
    """Run the lab safety filter against a predictions CSV."""
    _print_section("Safety Check")

    df = _load_predictions_csv(args.input)
    df = _normalize_for_safety_checks(df)
    out_dir = _resolve_output_dir(args.output_dir) if args.output_dir else None
    safe_mode = not args.no_safe_mode
    override = args.allow_override

    print(f"  Input          : {Path(args.input).resolve()}")
    print(f"  Safe mode      : {safe_mode}")
    print(f"  Human override : {override}")
    print(f"  Records        : {len(df)}")

    try:
        from export_for_lab import filter_safe_materials, load_hazard_config
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import export_for_lab: {exc}")

    config = load_hazard_config()
    blacklisted = config.get("blacklist_elements", config.get("blacklisted_elements", []))
    print(f"  Hazard config  : {len(blacklisted)} blacklisted element(s): {blacklisted}")

    safe_df, unsafe_df, review_df = filter_safe_materials(
        df, safe_mode=safe_mode, human_override=override
    )

    total = len(df)
    print()
    print("  Results")
    print(f"    Safe        : {len(safe_df):>5} / {total}")
    print(f"    Unsafe      : {len(unsafe_df):>5} / {total}")
    print(f"    Human review: {len(review_df):>5} / {total}")
    print()

    # Show reasons for unsafe materials (up to --max-show rows)
    max_show = args.max_show
    if not unsafe_df.empty and "safety_reason" in unsafe_df.columns:
        print(f"  Unsafe materials (showing up to {max_show}):")
        for _, row in unsafe_df.head(max_show).iterrows():
            formula = row.get("formula", row.get("Formula", "?"))
            reason = row.get("safety_reason", "unknown reason")
            print(f"    ✗  {formula}: {reason}")

    if not review_df.empty and "safety_warnings" in review_df.columns:
        reviewed_shown = min(max_show, len(review_df))
        print(f"\n  Human-review materials (showing up to {max_show}):")
        for _, row in review_df.head(max_show).iterrows():
            formula = row.get("formula", row.get("Formula", "?"))
            warnings = row.get("safety_warnings", "")
            print(f"    ⚠  {formula}: {warnings}")

    # Save detailed reports if --output-dir given
    if out_dir is not None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not safe_df.empty:
            p = out_dir / f"safety_pass_{ts}.csv"
            safe_df.to_csv(p, index=False)
            print(f"\n  [OK] Safe materials saved   → {p}")
        if not unsafe_df.empty:
            p = out_dir / f"safety_fail_{ts}.csv"
            unsafe_df.to_csv(p, index=False)
            print(f"  [OK] Unsafe materials saved → {p}")
        if not review_df.empty:
            p = out_dir / f"safety_review_{ts}.csv"
            review_df.to_csv(p, index=False)
            print(f"  [OK] Review materials saved → {p}")

    # Exit code 0 = all safe, 2 = some failed/need review
    if len(unsafe_df) > 0 or len(review_df) > 0:
        sys.exit(2)


# ---------------------------------------------------------------------------
# Command: export-lab
# ---------------------------------------------------------------------------

def cmd_export_lab(args: argparse.Namespace) -> None:
    """Generate the full safe-ish lab export (CSV + PDF) from a predictions CSV."""
    _print_section("Safe-Ish Lab Export")

    df = _load_predictions_csv(args.input)
    out_dir = _resolve_output_dir(args.output_dir)
    safe_mode = not args.no_safe_mode
    override = args.allow_override

    print(f"  Input          : {Path(args.input).resolve()}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Safe mode      : {safe_mode}")
    print(f"  Human override : {override}")
    print(f"  Records        : {len(df)}")

    # Optional: load ml_metrics from a JSON sidecar if provided
    ml_metrics = None
    if args.metrics_json:
        metrics_path = Path(args.metrics_json).resolve()
        if not metrics_path.exists():
            sys.exit(f"[ERROR] Metrics JSON not found: {metrics_path}")
        try:
            with open(metrics_path) as fh:
                ml_metrics = json.load(fh)
            print(f"  ML metrics     : loaded from {metrics_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load metrics JSON ({exc}). Using defaults.")

    try:
        from export_for_lab import export_for_lab
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import export_for_lab: {exc}")

    try:
        csv_path, pdf_path, safety_summary = export_for_lab(
            predictions_df=df,
            ml_metrics=ml_metrics,
            output_dir=str(out_dir),
            safe_mode=safe_mode,
            allow_human_override=override,
        )
    except Exception as exc:
        sys.exit(f"[ERROR] Export failed: {exc}")

    print()
    print("  Export complete:")
    print(f"    CSV  → {csv_path}")
    print(f"    PDF  → {pdf_path}")
    print()
    print("  Safety summary:")
    for k, v in safety_summary.items():
        print(f"    {k:<30}: {v}")


# ---------------------------------------------------------------------------
# Command: export-csv
# ---------------------------------------------------------------------------

def cmd_export_csv(args: argparse.Namespace) -> None:
    """Generate the generic all-materials analysis CSV from a predictions CSV."""
    _print_section("All-Materials Analysis CSV Export")

    import pandas as pd
    import tempfile
    import shutil
    from datetime import datetime

    df = _load_predictions_csv(args.input)
    out_dir = _resolve_output_dir(args.output_dir)

    print(f"  Input      : {Path(args.input).resolve()}")
    print(f"  Output dir : {out_dir}")
    print(f"  Records    : {len(df)}")

    # Mirror the column set used by prepare_csv_export() in gradio_app.py.
    export_columns = [
        "formula", "at%", "wt%", "feedstock_g_per_100g",
        "thermodynamic_stability_score", "thermodynamic_stability_category",
        "ensemble_probability", "ensemble_confidence", "ensemble_prediction",
        "in_distribution", "nn_distance", "calibration_status", "synthesis_priority_rank",
        "energy_above_hull", "formation_energy_per_atom", "density",
        "melting_point", "band_gap", "electronegativity", "atomic_radius",
        "nsites", "element_1", "element_2", "element_3",
        "composition_1", "composition_2", "composition_3",
    ]

    # Keep only columns that actually exist
    available = [c for c in export_columns if c in df.columns]
    missing = [c for c in export_columns if c not in df.columns]

    if missing:
        print(f"  [WARN] {len(missing)} column(s) absent from input CSV (will be omitted):")
        for m in missing:
            print(f"           • {m}")

    export_df = df[available].copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_materials_analysis_{ts}.csv"
    outpath = out_dir / filename

    export_df.to_csv(outpath, index=False)
    print(f"\n  [OK] Saved {len(export_df)} rows × {len(available)} cols → {outpath}")


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Show environment status: API key, hazard config, field mapping, schema."""
    _print_section("Environment Status")

    # 1. API key
    api_key = os.getenv("MP_API_KEY", "")
    print(f"\n  MP_API_KEY : {_mask_key(api_key) if api_key else '[not set]'}")

    # 2. Hazard config
    hazards_path = Path("hazards.yml")
    print(f"\n  Hazard config ({hazards_path}):")
    if hazards_path.exists():
        try:
            import yaml
            with open(hazards_path) as fh:
                config = yaml.safe_load(fh)
            # hazards.yml uses 'blacklist_elements' (see export_for_lab.load_hazard_config)
            blacklisted = config.get("blacklist_elements",
                                     config.get("blacklisted_elements", []))
            safe_threshold = config.get("safe_mode_threshold",
                                        config.get("synthesis_probability_threshold", "n/a"))
            print(f"    Blacklisted elements : {blacklisted}")
            print(f"    Safe mode threshold  : {safe_threshold}")
        except Exception as exc:
            print(f"    [WARN] Could not parse hazards.yml: {exc}")
    else:
        print("    [WARN] hazards.yml not found — default config will be used at runtime.")

    # 3. Field mapping
    print("\n  Field mapping (centralized_field_mapping.py):")
    try:
        from centralized_field_mapping import FIELD_ALIASES, CANONICAL_FIELDS
        print(f"    Canonical fields : {len(CANONICAL_FIELDS)}")
        print(f"    Alias groups     : {len(FIELD_ALIASES)}")
    except ImportError:
        try:
            from field_mapping_utils import REQUIRED_FIELDS_ML_CLASSIFIER
            print(f"    Required ML classifier fields: {REQUIRED_FIELDS_ML_CLASSIFIER}")
        except ImportError:
            print("    [WARN] Field mapping modules not importable.")

    # 4. ML feature schema
    schema_path = Path("feature_schema.yml")
    print(f"\n  ML feature schema ({schema_path}):")
    if schema_path.exists():
        try:
            import yaml
            with open(schema_path) as fh:
                schema = yaml.safe_load(fh)
            print(f"    Defined fields   : {len(schema)}")
            for fname, rules in list(schema.items())[:5]:
                rng = rules.get("range", "–")
                print(f"      {fname:<35} range={rng}")
            if len(schema) > 5:
                print(f"      … and {len(schema) - 5} more fields")
        except Exception as exc:
            print(f"    [WARN] Could not parse feature_schema.yml: {exc}")
    else:
        print("    [WARN] feature_schema.yml not found.")

    # 5. Key module importability
    print("\n  Module import checks:")
    modules = [
        ("materials_discovery_api", "MaterialsProjectClient"),
        ("synthesizability_predictor", "SynthesizabilityClassifier"),
        ("export_for_lab", "export_for_lab"),
        ("api_authentication_handler", "APIAuthenticationHandler"),
        ("field_mapping_utils", "validate_dataframe_consistency"),
    ]
    for mod, obj in modules:
        try:
            m = __import__(mod)
            if hasattr(m, obj):
                print(f"    [OK]   {mod}.{obj}")
            else:
                print(f"    [WARN] {mod} loaded but '{obj}' not found")
        except ImportError as exc:
            print(f"    [FAIL] {mod}: {exc}")

    # 6. Dataset presence
    print("\n  Local datasets:")
    for fname in ["materials_dataset.csv", "materials_project_raw_data.csv",
                  "materials_project_ml_features.csv", "experimental_workflow.csv"]:
        p = Path(fname)
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"    [OK] {fname:<45} {size_kb:>6} KB")
        else:
            print(f"    [--] {fname} (not present)")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=textwrap.dedent("""\
            Materials Discovery Workshop CLI
            ─────────────────────────────────
            Debug, validate, and export materials data from the command line.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # ---- validate-key ----
    p_vk = sub.add_parser("validate-key", help="Validate a Materials Project API key.")
    p_vk.add_argument("--api-key", metavar="KEY",
                      help="API key to test (default: MP_API_KEY env var).")
    p_vk.set_defaults(func=cmd_validate_key)

    # ---- fetch-materials ----
    p_fm = sub.add_parser("fetch-materials",
                           help="Fetch materials from the MP API and save to local CSVs.")
    p_fm.add_argument("--api-key", metavar="KEY",
                      help="MP API key (default: MP_API_KEY env var).")
    p_fm.add_argument("--elements", metavar="El1,El2",
                      help="Comma-separated element filter, e.g. 'Al,Ti'. "
                           "Omit to fetch all binary alloys.")
    p_fm.add_argument("--limit", type=int, default=200,
                      help="Maximum number of materials to fetch (default: 200).")
    p_fm.add_argument("--hull-max", type=float, default=0.1,
                      help="Maximum energy above hull in eV/atom (default: 0.1).")
    p_fm.add_argument("--output-dir", default=".",
                      help="Directory to save CSV files (default: current directory).")
    p_fm.set_defaults(func=cmd_fetch_materials)

    # ---- check-safety ----
    p_cs = sub.add_parser("check-safety",
                           help="Run the lab safety filter on a local predictions CSV.")
    p_cs.add_argument("input", metavar="INPUT_CSV",
                      help="Path to the predictions CSV file.")
    p_cs.add_argument("--no-safe-mode", action="store_true",
                      help="Disable conservative safe-mode thresholds.")
    p_cs.add_argument("--allow-override", action="store_true",
                      help="Allow human override for borderline materials.")
    p_cs.add_argument("--output-dir", metavar="DIR", default=None,
                      help="If supplied, save pass/fail/review CSVs to this directory.")
    p_cs.add_argument("--max-show", type=int, default=20,
                      help="Max number of failing materials to display (default: 20).")
    p_cs.set_defaults(func=cmd_check_safety)

    # ---- export-lab ----
    p_el = sub.add_parser("export-lab",
                           help="Generate the safe-ish lab CSV + PDF from a predictions CSV.")
    p_el.add_argument("input", metavar="INPUT_CSV",
                      help="Path to the predictions CSV file.")
    p_el.add_argument("--output-dir", default=".",
                      help="Directory to save the CSV and PDF (default: current directory).")
    p_el.add_argument("--no-safe-mode", action="store_true",
                      help="Disable conservative safe-mode thresholds.")
    p_el.add_argument("--allow-override", action="store_true",
                      help="Allow human override for borderline materials.")
    p_el.add_argument("--metrics-json", metavar="FILE",
                      help="Optional JSON file with ML performance metrics to embed in the PDF.")
    p_el.set_defaults(func=cmd_export_lab)

    # ---- export-csv ----
    p_ec = sub.add_parser("export-csv",
                           help="Generate the all-materials analysis CSV from a predictions CSV.")
    p_ec.add_argument("input", metavar="INPUT_CSV",
                      help="Path to the predictions CSV file.")
    p_ec.add_argument("--output-dir", default=".",
                      help="Directory to save the CSV (default: current directory).")
    p_ec.set_defaults(func=cmd_export_csv)

    # ---- status ----
    p_st = sub.add_parser("status",
                           help="Show environment status (API key, hazards, imports, datasets).")
    p_st.set_defaults(func=cmd_status)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Ensure the project root is on sys.path so imports resolve correctly
    # when the CLI is invoked from any working directory.
    project_root = str(Path(__file__).parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    args.func(args)


if __name__ == "__main__":
    main()
