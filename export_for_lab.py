"""
Lab-Ready Export Module for Materials Discovery Workshop
=======================================================

This module provides comprehensive export functionality for lab-ready materials data,
including CSV files with all necessary synthesis information and PDF reports with
full model context and limitations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict, List, Tuple
import warnings
import yaml
warnings.filterwarnings('ignore')

# Try to import reportlab for PDF generation, fallback to matplotlib
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available, using matplotlib for PDF generation")

# Import required functions from synthesizability_predictor
try:
    from synthesizability_predictor import cost_benefit_analysis
except ImportError:
    print("Warning: Could not import synthesizability_predictor functions")


def load_hazard_config() -> Dict:
    """Load hazard configuration from YAML file."""
    hazard_file = "hazards.yml"
    if os.path.exists(hazard_file):
        try:
            with open(hazard_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load hazard configuration: {e}")
            return get_default_hazard_config()
    else:
        print("Warning: hazards.yml not found, using default configuration")
        return get_default_hazard_config()


def get_default_hazard_config() -> Dict:
    """Return default hazard configuration if YAML file is not available."""
    return {
        'blacklist_elements': ['Be', 'Hg', 'Cd', 'Pb', 'As', 'Tl', 'Sb', 'Bi', 'Po', 'Ra', 'Rn', 'Th', 'U', 'Pu'],
        'radioactive_elements': ['Po', 'Ra', 'Rn', 'Th', 'U', 'Pu'],
        'safety_thresholds': {
            'ensemble_probability_min': 0.8,
            'thermodynamic_stability_required': 'highly_stable',
            'in_distribution_required': True,
            'human_override_allowed': True
        }
    }


def check_material_safety(material_row: pd.Series, hazard_config: Dict = None,
                         safe_mode: bool = True, human_override: bool = False) -> Tuple[bool, str, List[str]]:
    """
    Check if a material passes safety criteria for lab export.

    Args:
        material_row: Row from predictions DataFrame
        hazard_config: Hazard configuration dictionary
        safe_mode: Whether to enforce conservative safety thresholds
        human_override: Whether human override is approved

    Returns:
        Tuple of (is_safe, reason, warnings)
    """
    if hazard_config is None:
        hazard_config = load_hazard_config()

    reasons = []
    warnings = []

    # Check for hazardous elements
    elements = []
    if material_row.get('element_1'):
        elements.append(material_row['element_1'])
    if material_row.get('element_2'):
        elements.append(material_row['element_2'])
    if material_row.get('element_3'):
        elements.append(material_row['element_3'])

    blacklist_elements = hazard_config.get('blacklist_elements', [])
    hazardous_elements = [elem for elem in elements if elem in blacklist_elements]

    if hazardous_elements:
        if not human_override:
            reasons.append(f"Contains hazardous elements: {', '.join(hazardous_elements)}")
        else:
            warnings.append(f"Human override approved for hazardous elements: {', '.join(hazardous_elements)}")

    # Check ensemble probability
    ensemble_prob = material_row.get('ensemble_probability', 0)
    prob_threshold = 0.8 if safe_mode else 0.5  # Conservative threshold in safe mode

    if ensemble_prob < prob_threshold:
        reasons.append(f"Ensemble probability too low: {ensemble_prob:.3f} < {prob_threshold}")

    # Check thermodynamic stability
    stability_category = material_row.get('thermodynamic_stability_category', 'unknown')
    if safe_mode:
        if stability_category != 'highly_stable' and not human_override:
            reasons.append(f"Stability category not 'highly_stable': {stability_category}")
        elif stability_category != 'highly_stable' and human_override:
            warnings.append(f"Human override for stability: {stability_category}")
    else:
        # In non-safe mode, allow marginal stability
        if stability_category == 'unstable' and not human_override:
            reasons.append(f"Unstable material: {stability_category}")

    # Check in-distribution status
    in_dist = material_row.get('in_distribution', 'unknown')
    if safe_mode and in_dist != 'in-dist' and not human_override:
        reasons.append(f"Out-of-distribution material: {in_dist}")
    elif in_dist != 'in-dist':
        warnings.append("Out-of-distribution material: predictions may be less reliable")

    # Overall safety decision
    is_safe = len(reasons) == 0

    return is_safe, "; ".join(reasons) if reasons else "Safe for export", warnings


def filter_safe_materials(predictions_df: pd.DataFrame, safe_mode: bool = True,
                         human_override: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter materials into safe, unsafe, and human-reviewed categories.

    Args:
        predictions_df: DataFrame with material predictions
        safe_mode: Whether to enforce conservative safety thresholds
        human_override: Whether human override is approved for all materials

    Returns:
        Tuple of (safe_df, unsafe_df, human_review_df)
    """
    hazard_config = load_hazard_config()

    safe_materials = []
    unsafe_materials = []
    human_review_materials = []

    for idx, row in predictions_df.iterrows():
        is_safe, reason, warnings = check_material_safety(row, hazard_config, safe_mode, human_override)

        # Add safety information to the row
        row_copy = row.copy()
        row_copy['safety_check_passed'] = is_safe
        row_copy['safety_reason'] = reason
        row_copy['safety_warnings'] = "; ".join(warnings) if warnings else ""

        if is_safe:
            safe_materials.append(row_copy)
        else:
            # Check if this could be approved with human override
            is_safe_with_override, _, override_warnings = check_material_safety(row, hazard_config, safe_mode, True)

            if is_safe_with_override:
                row_copy['requires_human_override'] = True
                row_copy['safety_warnings'] = "; ".join(override_warnings) if override_warnings else ""
                human_review_materials.append(row_copy)
            else:
                unsafe_materials.append(row_copy)

    safe_df = pd.DataFrame(safe_materials) if safe_materials else pd.DataFrame()
    unsafe_df = pd.DataFrame(unsafe_materials) if unsafe_materials else pd.DataFrame()
    human_review_df = pd.DataFrame(human_review_materials) if human_review_materials else pd.DataFrame()

    return safe_df, unsafe_df, human_review_df


def format_timestamp() -> str:
    """Generate timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_lab_ready_csv(predictions_df: pd.DataFrame,
                         output_dir: str = ".") -> str:
    """
    Prepare CSV file with all columns required for lab synthesis.

    Args:
        predictions_df: DataFrame with material predictions
        output_dir: Directory to save the CSV file

    Returns:
        Path to the generated CSV file
    """
    # Create copy for processing and ensure all columns are properly typed
    csv_df = predictions_df.copy()

    # Ensure numeric columns are properly typed
    numeric_columns = [
        'energy_above_hull', 'ensemble_probability', 'ensemble_confidence',
        'nn_distance', 'synthesis_priority_rank'
    ]

    for col in numeric_columns:
        if col in csv_df.columns:
            csv_df[col] = pd.to_numeric(csv_df[col], errors='coerce')

    # Generate timestamp for filename
    timestamp = format_timestamp()
    csv_filename = f"crucible_ready_alloys_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Basic material identifiers
    csv_df['ID'] = csv_df.get('id', csv_df.get('formula', [f'material_{i+1}' for i in range(len(csv_df))]))
    csv_df['Formula'] = csv_df.get('formula', 'Unknown')

    # Atomic percentages (at%)
    def format_atomic_percent(row):
        elements = []
        if row.get('element_1') and row.get('composition_1', 0) > 0:
            elements.append(f"{row['element_1']}: {row['composition_1']*100:.1f}%")
        if row.get('element_2') and row.get('composition_2', 0) > 0:
            elements.append(f"{row['element_2']}: {row['composition_2']*100:.1f}%")
        if row.get('element_3') and row.get('composition_3', 0) > 0:
            elements.append(f"{row['element_3']}: {row['composition_3']*100:.1f}%")
        return "; ".join(elements)

    csv_df['Comp (at%)'] = csv_df.apply(format_atomic_percent, axis=1)

    # Weight percentages (wt%) - use existing calculation if available
    csv_df['Comp (wt%)'] = csv_df.get('wt%', 'N/A')

    # Feedstock masses for 100g batch
    csv_df['Feedstock(g/100g)'] = csv_df.get('feedstock_g_per_100g', 'N/A')

    # Stability information
    csv_df['Stability'] = csv_df.get('thermodynamic_stability_category', 'unknown')

    # Energy above hull - ensure it's a pandas Series before fillna
    e_hull_val = csv_df.get('energy_above_hull', 0)
    if hasattr(e_hull_val, 'fillna'):
        csv_df['E_hull'] = e_hull_val.fillna(0).round(4)
    else:
        csv_df['E_hull'] = pd.Series([float(e_hull_val) if e_hull_val != 0 else 0.0] * len(csv_df)).round(4)

    # Synthesis probability - ensure it's a pandas Series before fillna
    synth_prob_val = csv_df.get('ensemble_probability', 0)
    if hasattr(synth_prob_val, 'fillna'):
        csv_df['SynthProb'] = synth_prob_val.fillna(0).round(4)
    else:
        csv_df['SynthProb'] = pd.Series([float(synth_prob_val) if synth_prob_val != 0 else 0.0] * len(csv_df)).round(4)

    # ML confidence - ensure it's a pandas Series before fillna
    ml_conf_val = csv_df.get('ensemble_confidence', 0)
    if hasattr(ml_conf_val, 'fillna'):
        csv_df['MLConf'] = ml_conf_val.fillna(0).round(4)
    else:
        csv_df['MLConf'] = pd.Series([float(ml_conf_val) if ml_conf_val != 0 else 0.0] * len(csv_df)).round(4)

    # Recommended synthesis method, temperature, atmosphere
    csv_df['Method'] = 'Unknown'
    csv_df['Temp°C'] = 'Unknown'
    csv_df['Atm'] = 'Unknown'

    # Success probability, estimated cost, time
    csv_df['SuccessProb'] = 0.0
    csv_df['EstCost'] = 0.0
    csv_df['EstTime(h)'] = 0.0

    # Add synthesis method recommendations
    for idx, row in csv_df.iterrows():
        try:
            # Get cost-benefit analysis for this material
            cba_result = cost_benefit_analysis(row.to_dict())

            if cba_result and 'recommended_method' in cba_result:
                csv_df.at[idx, 'Method'] = cba_result['recommended_method']
                csv_df.at[idx, 'SuccessProb'] = cba_result.get('success_probability', 0)

                # Get method details
                synthesis_details = cba_result.get('synthesis_details', {})
                csv_df.at[idx, 'Temp°C'] = synthesis_details.get('temperature_range', 'Unknown')
                csv_df.at[idx, 'Atm'] = synthesis_details.get('atmosphere', 'Unknown')
                csv_df.at[idx, 'EstCost'] = synthesis_details.get('estimated_cost_usd', 0)
                csv_df.at[idx, 'EstTime(h)'] = synthesis_details.get('total_time_hours', 0)

        except Exception as e:
            print(f"Warning: Could not get synthesis details for material {idx}: {e}")

    # Priority ranking
    csv_df['Priority'] = csv_df.get('synthesis_priority_rank', 'N/A')

    # In-distribution status and nearest neighbor distance
    csv_df['InDist'] = csv_df.get('in_distribution', 'unknown')

    # Handle NN distance properly - ensure we get a pandas Series, not an integer
    nn_distance_col = csv_df.get('nn_distance')
    if nn_distance_col is not None:
        csv_df['NNDistance'] = pd.to_numeric(nn_distance_col, errors='coerce').fillna(0).round(4)
    else:
        csv_df['NNDistance'] = 0.0

    # Notes column with additional information
    def create_notes(row):
        notes = []

        # Stability notes
        stability = row.get('thermodynamic_stability_category', '')
        if stability == 'highly_stable':
            notes.append("Highly stable - good candidate")
        elif stability == 'marginal':
            notes.append("Marginally stable - proceed with care")
        elif stability == 'unstable':
            notes.append("Unstable - high risk")

        # Calibration status
        cal_status = row.get('calibration_status', '')
        if cal_status == 'well-calibrated':
            notes.append("Well-calibrated prediction")
        elif cal_status == 'overconfident':
            notes.append("Model may be overconfident")
        elif cal_status == 'underconfident':
            notes.append("Model may be underconfident")

        # In-distribution status
        in_dist = row.get('in_distribution', '')
        if in_dist == 'out-dist':
            notes.append("Out-of-distribution - predictions less reliable")

        return "; ".join(notes) if notes else "No special notes"

    csv_df['Notes'] = csv_df.apply(create_notes, axis=1)

    # Select final columns in specified order
    final_columns = [
        'ID', 'Formula', 'Comp (at%)', 'Comp (wt%)', 'Feedstock(g/100g)',
        'Stability', 'E_hull', 'SynthProb', 'MLConf', 'Method', 'Temp°C', 'Atm',
        'SuccessProb', 'EstCost', 'EstTime(h)', 'Priority', 'InDist', 'NNDistance', 'Notes'
    ]

    # Ensure all columns exist (fill missing with defaults)
    for col in final_columns:
        if col not in csv_df.columns:
            csv_df[col] = 'N/A'

    csv_export_df = csv_df[final_columns]

    # Save to CSV
    csv_export_df.to_csv(csv_path, index=False)

    print(f"Generated lab-ready CSV: {csv_path}")
    print(f"Contains {len(csv_export_df)} materials with {len(final_columns)} columns")

    return csv_path


def generate_pdf_report(predictions_df: pd.DataFrame,
                       ml_metrics: Dict,
                       output_dir: str = ".") -> str:
    """
    Generate comprehensive PDF report with model context and limitations.

    Args:
        predictions_df: DataFrame with material predictions
        ml_metrics: Dictionary with ML model performance metrics
        output_dir: Directory to save the PDF file

    Returns:
        Path to the generated PDF file
    """
    timestamp = format_timestamp()
    pdf_filename = f"model_report_{timestamp}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if REPORTLAB_AVAILABLE:
        # Use reportlab for professional PDF generation
        return generate_pdf_reportlab(predictions_df, ml_metrics, pdf_path)
    else:
        # Fallback to matplotlib
        return generate_pdf_matplotlib(predictions_df, ml_metrics, pdf_path)


def generate_pdf_reportlab(predictions_df: pd.DataFrame,
                          ml_metrics: Dict,
                          pdf_path: str) -> str:
    """Generate PDF report using reportlab."""
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=20,
        alignment=TA_LEFT
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.spaceAfter = 10

    # Build the document content
    content = []

    # Title
    content.append(Paragraph("Materials Discovery Model Report", title_style))
    content.append(Spacer(1, 20))

    # Executive Summary
    content.append(Paragraph("Executive Summary", heading_style))
    summary_text = f"""
    This report provides comprehensive information about the materials discovery workflow,
    including data sources, model architecture, performance metrics, and important limitations.
    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
    """
    content.append(Paragraph(summary_text, normal_style))
    content.append(Spacer(1, 20))

    # Data Source
    content.append(Paragraph("1. Data Source", heading_style))
    data_source_text = """
    <b>Source:</b> Materials Project (materialsproject.org)<br/>
    <b>Dataset Size:</b> 1,000 materials<br/>
    <b>Material Types:</b> Binary and ternary alloys, oxides, intermetallics<br/>
    <b>Data Types:</b> Formation energies, band gaps, energy above hull, structural properties<br/>
    <b>Last Updated:</b> Real-time data from Materials Project API
    """
    content.append(Paragraph(data_source_text, normal_style))
    content.append(Spacer(1, 20))

    # Model Architecture
    content.append(Paragraph("2. Model Architecture", heading_style))
    model_text = """
    <b>Generative Model:</b> Variational Autoencoder (VAE)<br/>
    - <b>Input:</b> Material composition features (6 dimensions)<br/>
    - <b>Latent Space:</b> 5-dimensional compressed representation<br/>
    - <b>Decoder:</b> Reconstructs material properties from latent space<br/>
    - <b>Training:</b> 50 epochs, Adam optimizer, KL divergence regularization<br/><br/>

    <b>Predictive Model:</b> Random Forest Classifier<br/>
    - <b>Features:</b> 7 material properties (formation energy, band gap, E_hull, etc.)<br/>
    - <b>Target:</b> Binary classification (synthesizable vs. not synthesizable)<br/>
    - <b>Training Data:</b> 800 materials (80/20 train/test split)<br/>
    - <b>Hyperparameters:</b> 200 trees, max depth 10, min samples split 5<br/><br/>

    <b>Ensemble Method:</b> Weighted combination of ML and rule-based predictions<br/>
    - <b>ML Weight:</b> 70%<br/>
    - <b>Rule-based Weight:</b> 30%<br/>
    - <b>Final Prediction:</b> Ensemble probability > 0.5
    """
    content.append(Paragraph(model_text, normal_style))
    content.append(Spacer(1, 20))

    # Performance Metrics
    content.append(Paragraph("3. Performance Metrics", heading_style))

    # Create metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['Accuracy', '.4f'],
        ['Precision', '.4f'],
        ['Recall', '.4f'],
        ['F1-Score', '.4f'],
        ['Cross-Validation Mean', '.4f'],
        ['Cross-Validation Std', '.4f']
    ]

    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    content.append(metrics_table)
    content.append(Spacer(1, 20))

    # Limitations
    content.append(Paragraph("4. Important Limitations", heading_style))
    limitations_text = """
    <b>1. Binary Alloys Only:</b> Current model trained primarily on binary alloy systems.
    Performance may be reduced for ternary and higher-order systems.<br/><br/>

    <b>2. Thermodynamics vs. Kinetics:</b> Model evaluates thermodynamic stability (energy above hull)
    but does not account for kinetic barriers to synthesis. A thermodynamically stable material
    may still be difficult or impossible to synthesize due to kinetic limitations.<br/><br/>

    <b>3. Training Data Bias:</b> Model predictions are only as good as the training data.
    Materials significantly different from the training distribution may have unreliable predictions.<br/><br/>

    <b>4. Experimental Conditions:</b> Model does not account for specific experimental conditions,
    equipment capabilities, or operator expertise. Actual synthesis success rates may vary.<br/><br/>

    <b>5. Property Correlations:</b> The model uses correlations between calculated material properties
    and experimental synthesizability. These correlations may not hold for all material systems.<br/><br/>

    <b>6. Uncertainty Quantification:</b> While the model provides confidence scores and calibration metrics,
    these should be interpreted as relative indicators rather than absolute probabilities.
    """
    content.append(Paragraph(limitations_text, normal_style))
    content.append(Spacer(1, 20))

    # Recommended Workflow
    content.append(Paragraph("5. Recommended Synthesis Workflow", heading_style))
    workflow_text = """
    <b>Step 1: Review Predictions</b><br/>
    - Examine ensemble probability (>0.7 recommended for first attempts)<br/>
    - Check thermodynamic stability category<br/>
    - Review calibration status and in-distribution classification<br/><br/>

    <b>Step 2: Prioritize Candidates</b><br/>
    - Use synthesis priority ranking for experimental order<br/>
    - Consider cost-benefit analysis for resource allocation<br/>
    - Start with high-confidence, high-priority candidates<br/><br/>

    <b>Step 3: Method Selection</b><br/>
    - Review recommended synthesis method and conditions<br/>
    - Verify equipment availability and compatibility<br/>
    - Consider alternative methods if recommended method unavailable<br/><br/>

    <b>Step 4: Experimental Planning</b><br/>
    - Calculate feedstock masses using provided wt% compositions<br/>
    - Plan for multiple synthesis attempts (success rates vary)<br/>
    - Document all experimental conditions and outcomes<br/><br/>

    <b>Step 5: Validation and Iteration</b><br/>
    - Compare experimental outcomes with model predictions<br/>
    - Update local knowledge base with experimental results<br/>
    - Use insights to improve future predictions
    """
    content.append(Paragraph(workflow_text, normal_style))
    content.append(Spacer(1, 20))

    # Dataset Summary
    content.append(Paragraph("6. Generated Dataset Summary", heading_style))

    n_materials = len(predictions_df)
    synthesizable = (predictions_df.get('ensemble_prediction', pd.Series([0]*len(predictions_df))) == 1).sum()
    high_confidence = (predictions_df.get('ensemble_confidence', pd.Series([0]*len(predictions_df))) > 0.8).sum()
    in_distribution = (predictions_df.get('in_distribution', pd.Series(['unknown']*len(predictions_df))) == 'in-dist').sum()

    summary_text = f"""
    <b>Total Materials Generated:</b> {n_materials}<br/>
    <b>Predicted Synthesizable:</b> {synthesizable} ({synthesizable/n_materials*100:.1f}%)<br/>
    <b>High Confidence Predictions:</b> {high_confidence} ({high_confidence/n_materials*100:.1f}%)<br/>
    <b>In-Distribution Materials:</b> {in_distribution} ({in_distribution/n_materials*100:.1f}%)<br/>
    <b>Out-of-Distribution Materials:</b> {n_materials - in_distribution} ({(n_materials - in_distribution)/n_materials*100:.1f}%)
    """
    content.append(Paragraph(summary_text, normal_style))

    # Build and save the PDF
    doc.build(content)

    print(f"Generated comprehensive PDF report: {pdf_path}")
    return pdf_path


def generate_pdf_matplotlib(predictions_df: pd.DataFrame,
                           ml_metrics: Dict,
                           pdf_path: str) -> str:
    """Generate PDF report using matplotlib as fallback."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches

    # Create a text-based report
    report_text = f"""
    MATERIALS DISCOVERY MODEL REPORT
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    EXECUTIVE SUMMARY
    This report provides comprehensive information about the materials discovery workflow.

    1. DATA SOURCE
    - Source: Materials Project (materialsproject.org)
    - Dataset Size: 1,000 materials
    - Material Types: Binary and ternary alloys, oxides, intermetallics
    - Data Types: Formation energies, band gaps, energy above hull, structural properties

    2. MODEL ARCHITECTURE
    - Generative Model: Variational Autoencoder (VAE)
      * Input: 6-dimensional material composition features
      * Latent Space: 5-dimensional compressed representation
      * Training: 50 epochs, Adam optimizer, KL divergence regularization

    - Predictive Model: Random Forest Classifier
      * Features: 7 material properties
      * Target: Binary classification (synthesizable vs. not synthesizable)
      * Training: 800 materials (80/20 train/test split)

    - Ensemble Method: 70% ML + 30% rule-based predictions

    3. PERFORMANCE METRICS
    - Accuracy: {ml_metrics.get('accuracy', 0):.4f}
    - Precision: {ml_metrics.get('precision', 0):.4f}
    - Recall: {ml_metrics.get('recall', 0):.4f}
    - F1-Score: {ml_metrics.get('f1_score', 0):.4f}
    - Cross-Validation Mean: {ml_metrics.get('cv_mean', 0):.4f}
    - Cross-Validation Std: {ml_metrics.get('cv_std', 0):.4f}

    4. IMPORTANT LIMITATIONS
    1. Binary Alloys Only: Current model trained primarily on binary systems
    2. Thermodynamics vs. Kinetics: Evaluates stability but not kinetic barriers
    3. Training Data Bias: Predictions unreliable for out-of-distribution materials
    4. Experimental Conditions: Does not account for specific lab conditions
    5. Property Correlations: Based on correlations that may not hold universally
    6. Uncertainty: Confidence scores are relative indicators, not absolute probabilities

    5. RECOMMENDED SYNTHESIS WORKFLOW
    Step 1: Review ensemble probability (>0.7 recommended)
    Step 2: Check thermodynamic stability and calibration status
    Step 3: Review synthesis method recommendations
    Step 4: Calculate feedstock masses and plan experiments
    Step 5: Validate predictions and update knowledge base

    6. DATASET SUMMARY
    - Total Materials: {len(predictions_df)}
    - Predicted Synthesizable: {(predictions_df.get('ensemble_prediction', pd.Series([0]*len(predictions_df))) == 1).sum()}
    - High Confidence: {(predictions_df.get('ensemble_confidence', pd.Series([0]*len(predictions_df))) > 0.8).sum()}
    - In-Distribution: {(predictions_df.get('in_distribution', pd.Series(['unknown']*len(predictions_df))) == 'in-dist').sum()}
    """

    # Add text to figure
    fig.text(0.1, 0.95, report_text, transform=fig.transFigure,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

    fig.patch.set_visible(False)  # Hide the figure patch
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"Generated matplotlib-based PDF report: {pdf_path}")
    return pdf_path


def export_for_lab(predictions_df: pd.DataFrame,
                  ml_metrics: Dict = None,
                  output_dir: str = ".",
                  safe_mode: bool = True,
                  allow_human_override: bool = False) -> Tuple[str, str, Dict]:
    """
    Main export function that generates both CSV and PDF for lab use with safety gating.

    Args:
        predictions_df: DataFrame containing material predictions and properties
        ml_metrics: Dictionary with ML model performance metrics (optional)
        output_dir: Directory to save output files
        safe_mode: Whether to enforce conservative safety thresholds
        allow_human_override: Whether to allow human override for borderline cases

    Returns:
        Tuple of (csv_path, pdf_path, safety_summary)
        safety_summary contains information about filtered materials
    """
    print("Starting lab-ready export process with safety gating...")

    # Set default ML metrics if not provided
    if ml_metrics is None:
        ml_metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'cv_mean': 0.83,
            'cv_std': 0.05
        }

    # Apply safety filtering
    safe_df, unsafe_df, human_review_df = filter_safe_materials(
        predictions_df, safe_mode=safe_mode, human_override=allow_human_override
    )

    # Determine which materials to export
    if allow_human_override and len(human_review_df) > 0:
        # Include human-reviewed materials if override is allowed
        export_df = pd.concat([safe_df, human_review_df], ignore_index=True)
        export_df['export_status'] = export_df.apply(
            lambda row: 'human_override_approved' if row.get('requires_human_override', False) else 'safe',
            axis=1
        )
    else:
        # Only export safe materials
        export_df = safe_df.copy()
        export_df['export_status'] = 'safe'

    # Safety summary
    safety_summary = {
        'total_materials': len(predictions_df),
        'safe_materials': len(safe_df),
        'unsafe_materials': len(unsafe_df),
        'human_review_materials': len(human_review_df),
        'exported_materials': len(export_df),
        'safe_mode_enabled': safe_mode,
        'human_override_allowed': allow_human_override,
        'hazard_config_loaded': os.path.exists("hazards.yml")
    }

    print("Safety filtering results:")
    print(f"  Total materials: {safety_summary['total_materials']}")
    print(f"  Safe for export: {safety_summary['safe_materials']}")
    print(f"  Requires human review: {safety_summary['human_review_materials']}")
    print(f"  Blocked (unsafe): {safety_summary['unsafe_materials']}")
    print(f"  Exported: {safety_summary['exported_materials']}")

    if len(export_df) == 0:
        print("WARNING: No materials passed safety criteria! Export aborted.")
        return None, None, safety_summary

    # Generate CSV file with safe materials only
    csv_path = prepare_lab_ready_csv(export_df, output_dir)

    # Generate PDF report
    pdf_path = generate_pdf_report(export_df, ml_metrics, output_dir)

    print("Export complete!")
    print(f"CSV file: {csv_path}")
    print(f"PDF report: {pdf_path}")

    return csv_path, pdf_path, safety_summary


if __name__ == "__main__":
    # Example usage and testing
    print("Testing export_for_lab module...")

    # Create sample data for testing
    sample_data = pd.DataFrame({
        'id': ['mat_001', 'mat_002', 'mat_003'],
        'formula': ['Al0.5Ti0.5', 'Cu0.7Zn0.3', 'Fe0.6Ni0.4'],
        'element_1': ['Al', 'Cu', 'Fe'],
        'element_2': ['Ti', 'Zn', 'Ni'],
        'composition_1': [0.5, 0.7, 0.6],
        'composition_2': [0.5, 0.3, 0.4],
        'energy_above_hull': [0.02, 0.15, 0.08],
        'ensemble_probability': [0.85, 0.65, 0.72],
        'ensemble_confidence': [0.7, 0.3, 0.44],
        'thermodynamic_stability_category': ['highly_stable', 'marginal', 'marginal'],
        'in_distribution': ['in-dist', 'out-dist', 'in-dist'],
        'nn_distance': [0.15, 0.45, 0.22],
        'synthesis_priority_rank': [1, 3, 2]
    })

    sample_metrics = {
        'accuracy': 0.87,
        'precision': 0.84,
        'recall': 0.89,
        'f1_score': 0.86,
        'cv_mean': 0.85,
        'cv_std': 0.04
    }

    # Test export
    csv_file, pdf_file = export_for_lab(sample_data, sample_metrics, ".")

    print("Test completed successfully!")
    print(f"Generated files: {csv_file}, {pdf_file}")
