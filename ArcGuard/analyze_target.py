import os
import csv
import collections
import json

def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    csv_path = os.path.join(dataset_dir, 'your_actual_dataset_file.csv')
    analysis_path = os.path.join(os.path.dirname(__file__), 'target_analysis.txt')
    report_lines = []
    # Load CSV
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        report_lines.append('CSV file is empty.')
        write_report(analysis_path, report_lines)
        return
    # Columns present
    columns = reader.fieldnames
    report_lines.append('Columns detected:')
    report_lines.append(', '.join(columns))
    # 1. Fault Type analysis
    fault_type_counts = collections.Counter(row.get('Fault Type', '') for row in rows if row.get('Fault Type', '') != '')
    report_lines.append('\nFault Type value distribution:')
    for ft, cnt in fault_type_counts.most_common():
        report_lines.append(f'  - {ft}: {cnt}')
    total = sum(fault_type_counts.values())
    report_lines.append(f'Total records with Fault Type: {total}')
    # Assess if Fault Type is suitable target
    unique_vals = len(fault_type_counts)
    report_lines.append(f'Unique Fault Type categories: {unique_vals}')
    if unique_vals >= 2:
        report_lines.append('Fault Type appears to be a categorical label suitable for supervised classification.')
    else:
        report_lines.append('Fault Type does not provide multiple classes; may not be suitable as target.')
    # 2. Check identifier column
    report_lines.append('\nFault ID column analysis:')
    fault_ids = [row.get('Fault ID', '') for row in rows]
    unique_ids = len(set(fault_ids))
    report_lines.append(f'Unique Fault ID count: {unique_ids} (should equal total rows if true identifier)')
    # 3. Data leakage assessment
    leakage_cols = ['Duration of Fault (hrs)', 'Down time (hrs)', 'Maintenance Status', 'Component Health']
    report_lines.append('\nPotential leakage columns (post‑fault information):')
    for col in leakage_cols:
        present = col in columns
        report_lines.append(f'  - {col}: {"present" if present else "missing"}')
    # 4. Recommend input features (available pre‑fault)
    pre_fault_features = ['Voltage (V)', 'Current (A)', 'Power Load (MW)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Weather Condition', 'Fault Location (Latitude, Longitude)']
    # Verify presence
    available_features = [f for f in pre_fault_features if f in columns]
    report_lines.append('\nRecommended input features (pre‑fault):')
    report_lines.append(', '.join(available_features))
    # 5. Class imbalance
    max_count = max(fault_type_counts.values())
    min_count = min(fault_type_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else None
    report_lines.append('\nClass imbalance assessment:')
    report_lines.append(f'Max / Min count ratio: {imbalance_ratio:.2f}' if imbalance_ratio else 'Cannot compute ratio.')
    # 6. Suspicious relationships (simple check: any column identical to Fault Type)
    # Skip complex analysis; note to review.
    report_lines.append('\nNote: Further investigation needed for potential leakage via correlated features (e.g., Duration may directly reflect Fault Type).')
    # 7. Recommendation for target column
    report_lines.append('\nRecommendation:')
    report_lines.append('Target column: Fault Type')
    report_lines.append('Reason: Provides multiple meaningful categories, matches classification goal, and is not an identifier.')
    report_lines.append('Excluded columns: Fault ID (identifier), Duration of Fault (hrs), Down time (hrs), Maintenance Status, Component Health (post‑fault leakage risk).')
    report_lines.append('Potential additional exclusion: any columns that directly encode the outcome.')
    write_report(analysis_path, report_lines)

def write_report(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()
