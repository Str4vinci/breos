import argparse
import pandas as pd
import os
import sys
import re

def main():
    parser = argparse.ArgumentParser(description="Normalize an RLP to a target yearly consumption.")
    parser.add_argument("input_csv", help="Path to the input profile CSV.")
    parser.add_argument("target_kwh", type=float, help="Target yearly consumption in kWh.")
    parser.add_argument("--column", help="The column to scale. Required if the CSV has multiple data columns.")
    parser.add_argument("--output", help="Optional path to output the scaled profile. If not provided, it saves as `<basename>_<target_kwh>kwh.csv`.")
    parser.add_argument("--date_format", help="Optional format for datetime parsing, e.g. '%d/%m/%Y %H:%M'")
    
    args = parser.parse_args()
    
    # Read the CSV
    try:
        # Assuming first column is the datetime index
        df = pd.read_csv(args.input_csv, index_col=0)
    except Exception as e:
        print(f"Error reading {args.input_csv}: {e}")
        sys.exit(1)
        
    try:
        if args.date_format:
            df.index = pd.to_datetime(df.index, format=args.date_format)
        else:
            try:
                # pandas usually infers standard ISO format safely
                df.index = pd.to_datetime(df.index)
            except Exception:
                # Fallback for ambiguous dayfirst formats
                df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
    except Exception as e:
        print(f"Error parsing datetime index: {e}")
        sys.exit(1)
    
    # Determine which column to scale
    if args.column:
        if args.column not in df.columns:
            print(f"Error: Column '{args.column}' not found in {args.input_csv}.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        target_col = args.column
    else:
        if len(df.columns) == 1:
            target_col = df.columns[0]
            print(f"Only one column found. Using '{target_col}'.")
        else:
            print(f"Error: Multiple columns found in {args.input_csv}. Please specify one using --column.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
            
    # Calculate regular interval in hours
    try:
        if df.index.tz is not None:
             df.index = df.index.tz_localize(None) # Removing timezone to avoid issues with arithmetic
        diffs = df.index.to_series().diff().dropna()
        interval_hours = diffs.mode()[0].total_seconds() / 3600.0
    except Exception as e:
        print(f"Warning: Could not determine valid time interval from index. Assuming 15-min (0.25 hrs).")
        interval_hours = 0.25
        
    print(f"Detected time interval: {interval_hours:.4f} hours.")
    
    # Detect units and calculate current consumption
    col_upper = target_col.upper()
    total_sum = df[target_col].sum()
    
    if "KWH" in col_upper:
        current_kwh = total_sum
        print(f"Detected unit: kWh. Current sum: {current_kwh:.2f} kWh")
    elif "WH" in col_upper:
        current_kwh = total_sum / 1000.0
        print(f"Detected unit: Wh. Current sum: {current_kwh:.2f} kWh")
    elif "KW" in col_upper:
        current_kwh = total_sum * interval_hours
        print(f"Detected unit: kW (Power). Current sum: {current_kwh:.2f} kWh")
    elif "W" in col_upper:
        current_kwh = (total_sum * interval_hours) / 1000.0
        print(f"Detected unit: W (Power). Current sum: {current_kwh:.2f} kWh")
    else:
        print(f"Warning: Could not detect unit from column name '{target_col}'. Assuming W (Power).")
        current_kwh = (total_sum * interval_hours) / 1000.0
        print(f"Current sum (assuming W): {current_kwh:.2f} kWh")
        
    if current_kwh <= 0:
        print("Error: Current yearly consumption is 0 or negative. Cannot scale.")
        sys.exit(1)
        
    factor = args.target_kwh / current_kwh
    print(f"Scaling factor: {factor:.6f}")
    
    # Scale profile
    scaled_series = df[target_col] * factor
    
    # Select just the scaled profile into a new DataFrame
    output_df = pd.DataFrame({target_col: scaled_series})
    
    # Check new sum
    if "KWH" in col_upper:
        new_kwh = output_df[target_col].sum()
    elif "WH" in col_upper:
        new_kwh = output_df[target_col].sum() / 1000.0
    elif "KW" in col_upper:
        new_kwh = output_df[target_col].sum() * interval_hours
    else:
        new_kwh = (output_df[target_col].sum() * interval_hours) / 1000.0
        
    print(f"Verification - New yearly consumption: {new_kwh:.2f} kWh")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_dir = os.path.dirname(args.input_csv)
        base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        # Keep it clean
        output_name = f"{base_name}_{int(args.target_kwh)}kwh.csv"
        output_path = os.path.join(base_dir, output_name)
        
    output_df.to_csv(output_path)
    print(f"Successfully saved scaled profile to: {output_path}")

if __name__ == "__main__":
    main()
