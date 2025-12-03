# -*- coding: utf-8 -*-
"""
Data Preprocessing Program
- Filter only electronics data from original CSV
- Save as new CSV file
"""

import pandas as pd

def filter_electronics_data(input_file, output_file):
    """
    Filter only electronics data and save as new CSV

    Parameters:
    -----------
    input_file : str
        Original CSV file path
    output_file : str
        Output CSV file path
    """
    print("="*80)
    print("Electronics Data Filtering Program")
    print("="*80)

    # Step 1: Load original data
    print(f"\n[Step 1] Loading original data...")
    df = pd.read_csv(input_file)
    print(f"Done")
    print(f"  - Total data size: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Data info
    print(f"\n[Data Information]")
    print(f"  - Columns: {list(df.columns)}")
    print(f"\n  - Categories:")
    category_list = df['업종분류'].unique()
    for idx, category in enumerate(category_list, 1):
        count = len(df[df['업종분류'] == category])
        print(f"    {idx}. {category}: {count:,} records")

    # Step 2: Filter electronics data only
    print(f"\n[Step 2] Filtering electronics data...")
    electronics_df = df[df['업종분류'] == '가전제품'].copy()
    print(f"Done")
    print(f"  - Electronics data: {electronics_df.shape[0]:,} rows x {electronics_df.shape[1]} columns")
    print(f"  - Percentage: {electronics_df.shape[0] / df.shape[0] * 100:.2f}%")

    # Step 3: Filtered data statistics
    print(f"\n[Step 3] Filtered data statistics")
    print(f"  - Period: {electronics_df['날짜'].min()} ~ {electronics_df['날짜'].max()}")
    print(f"  - Total sales amount: {electronics_df['매출금액'].sum():,.0f} KRW")
    print(f"  - Total sales count: {electronics_df['매출건수'].sum():,}")
    print(f"  - Average sales amount: {electronics_df['매출금액'].mean():,.0f} KRW")
    print(f"  - Average sales count: {electronics_df['매출건수'].mean():.1f}")

    # Gender distribution
    print(f"\n  - Gender distribution:")
    for gender in electronics_df['성별'].unique():
        count = len(electronics_df[electronics_df['성별'] == gender])
        total_sales = electronics_df[electronics_df['성별'] == gender]['매출금액'].sum()
        print(f"    {gender}: {count:,} records (Sales: {total_sales:,.0f} KRW)")

    # Age distribution
    print(f"\n  - Age distribution:")
    for age in sorted(electronics_df['연령대'].unique()):
        count = len(electronics_df[electronics_df['연령대'] == age])
        total_sales = electronics_df[electronics_df['연령대'] == age]['매출금액'].sum()
        print(f"    {age}0s: {count:,} records (Sales: {total_sales:,.0f} KRW)")

    # Step 4: Save to CSV
    print(f"\n[Step 4] Saving to CSV...")
    electronics_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Done: {output_file}")

    print("\n" + "="*80)
    print("Electronics Data Filtering Completed!")
    print("="*80)
    print(f"\nGenerated file:")
    print(f"  - {output_file}")
    print(f"  - {electronics_df.shape[0]:,} rows x {electronics_df.shape[1]} columns")
    print("\n" + "="*80)

    return electronics_df


if __name__ == "__main__":
    # File paths
    input_file = "card_gyeonggi_202503.csv"
    output_file = "electronics_sales_data.csv"

    # Filter and save electronics data
    electronics_data = filter_electronics_data(input_file, output_file)

    print(f"\nNext steps:")
    print(f"  1. {output_file} has been created.")
    print(f"  2. You can use this file for machine learning/deep learning analysis.")
    print(f"  3. To use this file in A.py, change: csv_file = '{output_file}'")
