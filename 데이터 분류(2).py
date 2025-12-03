import pandas as pd

def select_columns(input_file, output_file):
    print("="*80)
    print("Column Selection Program")
    print("="*80)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print("Done")
    print(f"  - Original data size: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Step 2: Select only necessary columns
    print("\n[Step 2] Selecting columns...")
    print("  - Columns to keep: Date, Sales Amount, Sales Count")

    # Get column names by index
    col_date = df.columns[0]      # First column (Date)
    col_amount = df.columns[5]    # 6th column (Sales Amount)
    col_count = df.columns[6]     # 7th column (Sales Count)

    # Select 3 columns only
    selected_df = df[[col_date, col_amount, col_count]].copy()

    print("Done")
    print(f"  - New data size: {selected_df.shape[0]:,} rows x {selected_df.shape[1]} columns")

    # Step 3: Data statistics
    print("\n[Step 3] Data statistics")
    print(f"  - Date range: {selected_df.iloc[:, 0].min()} ~ {selected_df.iloc[:, 0].max()}")
    print(f"  - Total sales amount: {selected_df.iloc[:, 1].sum():,.0f} KRW")
    print(f"  - Total sales count: {selected_df.iloc[:, 2].sum():,}")
    print(f"  - Average sales amount: {selected_df.iloc[:, 1].mean():,.0f} KRW")
    print(f"  - Average sales count: {selected_df.iloc[:, 2].mean():.1f}")

    # Show first 10 rows
    print("\n[First 10 rows preview]")
    print(selected_df.head(10))

    # Step 4: Save to CSV
    print("\n[Step 4] Saving to CSV...")
    selected_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Done: {output_file}")

    print("\n" + "="*80)
    print("Column Selection Completed!")
    print("="*80)
    print(f"\nGenerated file:")
    print(f"  - {output_file}")
    print(f"  - {selected_df.shape[0]:,} rows x {selected_df.shape[1]} columns")
    print("\n" + "="*80)

    return selected_df


if __name__ == "__main__":
    # File paths
    input_file = "electronics_sales_data.csv"
    output_file = "electronics_final.csv"

    # Select columns and save
    final_data = select_columns(input_file, output_file)

    print("\nNext steps:")
    print(f"  1. {output_file} has been created.")
    print(f"  2. This file contains only 3 columns: Date, Sales Amount, Sales Count")
    print(f"  3. To use this file in A.py, change: csv_file = '{output_file}'")
