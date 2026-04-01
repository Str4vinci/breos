
import pandas as pd
import os

# Define paths
results_dir = 'results/tou_complete_optimization'
input_file = os.path.join(results_dir, 'all_results.csv')

# Check if file exists
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

# Load data
df = pd.read_csv(input_file)

# Columns to display for the optimal solution
display_cols = [
    'Consumption_kWh', 'N_Modules', 'PV_kWp', 'Battery_kWh', 
    'Detailed_Strategy', 'Tariff', 'Cycle', 
    'NPV_Eur', 'Net_Cost_Eur', 'Grid_Independence_%', 
    'Breakeven_Year', 'CAPEX_Eur'
]

# Adjust column names if needed (strategies might be in 'Strategy' column)
if 'Detailed_Strategy' not in df.columns and 'Strategy' in df.columns:
    df['Detailed_Strategy'] = df['Strategy']

# Group by Consumption
consumptions = df['Consumption_kWh'].unique()
consumptions.sort()

print(f"Found {len(consumptions)} consumption profiles: {consumptions}")

results_summary = []

for cons in consumptions:
    subset = df[df['Consumption_kWh'] == cons]
    
    # Save split file
    output_filename = os.path.join(results_dir, f'results_{int(cons)}.csv')
    subset.to_csv(output_filename, index=False)
    print(f"Saved {len(subset)} rows to {output_filename}")
    
    # Find Ideal Solution (Max NPV)
    # Filter out where NPV might be NaN or invalid if necessary
    valid_subset = subset.dropna(subset=['NPV_Eur'])
    
    if not valid_subset.empty:
        best_npv_idx = valid_subset['NPV_Eur'].idxmax()
        best_sol = valid_subset.loc[best_npv_idx]
        
        results_summary.append(best_sol)
        
        print(f"\n--- Ideal Solution for {cons} kWh ---")
        # Handle cases where some columns might be missing
        cols = [c for c in display_cols if c in df.columns]
        print(best_sol[cols].to_string())
    else:
        print(f"No valid NPV results for {cons} kWh")

# Summary DataFrame
summary_df = pd.DataFrame(results_summary)
print("\n=== Overall Summary of Ideal Solutions ===")
cols = [c for c in display_cols if c in summary_df.columns]
print(summary_df[cols].to_string(index=False))

