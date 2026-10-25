import os
import csv
import sys
import collections
import math

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    report_path = os.path.join(os.path.dirname(__file__), 'dataset_report.txt')
    report_lines = []
    csv_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        report_lines.append('No CSV files found in dataset folder.')
        write_report(report_path, report_lines)
        return
    for csv_file in csv_files:
        file_path = os.path.join(dataset_dir, csv_file)
        report_lines.append(f'## Inspecting {csv_file}')
        try:
            with open(file_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                columns = reader.fieldnames or []
        except Exception as e:
            report_lines.append(f'Error reading file: {e}')
            continue
        if not rows:
            report_lines.append('File is empty or contains only header.')
            continue
        report_lines.append(f'Rows: {len(rows)}, Columns: {len(columns)}')
        report_lines.append('Column names: ' + ', '.join(columns))
        # Missing values per column
        missing_counts = {col: 0 for col in columns}
        for row in rows:
            for col in columns:
                if row[col] is None or row[col] == '':
                    missing_counts[col] += 1
        total_missing = sum(missing_counts.values())
        if total_missing:
            report_lines.append('Missing values per column:')
            for col, cnt in missing_counts.items():
                if cnt:
                    report_lines.append(f'  - {col}: {cnt}')
        else:
            report_lines.append('No missing values.')
        # Duplicate rows
        seen = set()
        dup = 0
        for row in rows:
            row_tuple = tuple(row.items())
            if row_tuple in seen:
                dup += 1
            else:
                seen.add(row_tuple)
        report_lines.append(f'Duplicate rows: {dup}')
        # Identify label column
        possible_label_cols = [c for c in columns if c.lower() in ('label', 'target', 'class', 'target_label')]
        if possible_label_cols:
            label_col = possible_label_cols[0]
            report_lines.append(f'Detected label column: {label_col}')
            # Class distribution
            counter = collections.Counter(row[label_col] for row in rows if row[label_col] != '')
            report_lines.append('Class distribution:')
            for cls, cnt in counter.most_common():
                report_lines.append(f'  - {cls}: {cnt}')
        else:
            report_lines.append('No obvious label column detected.')
        # Numeric column statistics
        numeric_cols = []
        numeric_data = {}
        for col in columns:
            values = [safe_float(row[col]) for row in rows]
            numeric_vals = [v for v in values if v is not None]
            if numeric_vals:
                numeric_cols.append(col)
                numeric_data[col] = numeric_vals
        if numeric_cols:
            report_lines.append('Numeric column summary statistics:')
            for col in numeric_cols:
                vals = numeric_data[col]
                count = len(vals)
                mean = sum(vals) / count if count else 0
                var = sum((x - mean) ** 2 for x in vals) / count if count else 0
                std = math.sqrt(var)
                min_val = min(vals)
                max_val = max(vals)
                report_lines.append(f'  - {col}: count={count}, mean={mean:.3f}, std={std:.3f}, min={min_val}, max={max_val}')
        else:
            report_lines.append('No numeric columns detected.')
        report_lines.append('')
    write_report(report_path, report_lines)

def write_report(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()
