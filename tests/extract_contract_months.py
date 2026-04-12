"""
Extract contract months from CLC RAD/NON data by detecting ratio jumps.

For each contract with both RAD and NON data:
1. Compute ratio = RAD_Close / NON_Close
2. Detect ratio jumps (roll dates)
3. Use roll_rules_corrected.json to determine delivery month from roll date
4. Output extracted contract months

Usage:
    python tests/extract_contract_months.py
"""

import pandas as pd
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / 'data' / 'CLC'
CONFIG_FILE = _PROJECT_ROOT / 'config' / 'roll_rules_corrected.json'

MONTHS_MAP = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
              7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']


def load_roll_rules():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def build_symbol_to_rule_type(roll_rules):
    """Map symbol -> rule_type string (e.g. 'MPDM_11', 'DM_8')"""
    symbol_to_rule = {}
    for rule_type, rule_data in roll_rules.items():
        for symbol in rule_data.get('symbols', []):
            symbol_to_rule[symbol] = rule_type
    return symbol_to_rule


def detect_roll_dates(rad_df, non_df, min_ratio_change_pct=0.005):
    """
    Detect roll dates by finding jumps in RAD_Close / NON_Close ratio.
    
    Returns list of (date_index, date, ratio_before, ratio_after) tuples.
    """
    # Merge on Date
    merged = pd.merge(
        non_df[['Date', 'Close']].rename(columns={'Close': 'NON_Close'}),
        rad_df[['Date', 'Close']].rename(columns={'Close': 'RAD_Close'}),
        on='Date', how='inner'
    )
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    # Filter out zero/negative prices
    merged = merged[(merged['NON_Close'] > 0) & (merged['RAD_Close'] > 0)].reset_index(drop=True)
    
    if len(merged) < 10:
        return []
    
    merged['ratio'] = merged['RAD_Close'] / merged['NON_Close']
    
    # Detect jumps: pct change in ratio from previous day
    merged['ratio_pct_change'] = merged['ratio'].pct_change().abs()
    
    # Find significant jumps (roll events)
    jumps = merged[merged['ratio_pct_change'] > min_ratio_change_pct].copy()
    
    rolls = []
    for idx, row in jumps.iterrows():
        rolls.append({
            'date': row['Date'],
            'ratio_before': merged.loc[idx - 1, 'ratio'] if idx > 0 else None,
            'ratio_after': row['ratio'],
            'pct_change': row['ratio_pct_change']
        })
    
    return rolls


def get_delivery_month_from_roll_date(roll_date, rule_type):
    """
    Given a roll date and rule type, determine the delivery month.
    
    MPDM_X: roll is in month BEFORE delivery month → delivery = roll_month + 1
    DM_X: roll is IN delivery month → delivery = roll_month  
    THUR_PRIOR_2ND_FRI_OF_DM: roll is IN delivery month → delivery = roll_month
    """
    roll_month = roll_date.month
    
    if rule_type and rule_type.startswith('MPDM'):
        # Month previous to delivery month
        delivery_month = roll_month + 1 if roll_month < 12 else 1
    else:
        # DM or THUR_PRIOR: roll is in delivery month
        delivery_month = roll_month
    
    return delivery_month


def analyze_all_contracts():
    """Analyze all contracts and extract delivery months."""
    roll_rules = load_roll_rules()
    symbol_to_rule = build_symbol_to_rule_type(roll_rules)
    
    # Find all RAD files
    rad_files = sorted(DATA_DIR.glob('*_RAD.CSV'))
    
    results = []
    
    for rad_file in rad_files:
        symbol = rad_file.stem.replace('_RAD', '')
        non_file = DATA_DIR / f'{symbol}_NON.CSV'
        
        if not non_file.exists():
            continue
        
        rule_type = symbol_to_rule.get(symbol)
        
        # Load data
        try:
            rad = pd.read_csv(rad_file, names=COLS)
            non = pd.read_csv(non_file, names=COLS)
            rad['Date'] = pd.to_datetime(rad['Date'])
            non['Date'] = pd.to_datetime(non['Date'])
        except Exception as e:
            print(f"  {symbol}: Error loading data: {e}")
            continue
        
        # Detect rolls
        rolls = detect_roll_dates(rad, non)
        
        if len(rolls) == 0:
            results.append({
                'symbol': symbol,
                'rule_type': rule_type,
                'roll_count': 0,
                'contract_months': '',
                'delivery_months_num': '',
                'sample_roll_dates': '',
                'data_start': non['Date'].min(),
                'data_end': non['Date'].max(),
            })
            continue
        
        # Extract delivery months
        delivery_months = set()
        sample_dates = []
        for roll in rolls:
            dm = get_delivery_month_from_roll_date(roll['date'], rule_type)
            delivery_months.add(dm)
            if len(sample_dates) < 5:
                sample_dates.append(roll['date'].strftime('%Y-%m-%d'))
        
        # Convert to letter codes, sorted by month number
        dm_sorted = sorted(delivery_months)
        dm_letters = ','.join(MONTHS_MAP[m] for m in dm_sorted)
        dm_nums = ','.join(str(m) for m in dm_sorted)
        
        results.append({
            'symbol': symbol,
            'rule_type': rule_type or 'UNKNOWN',
            'roll_count': len(rolls),
            'contract_months': dm_letters,
            'delivery_months_num': dm_nums,
            'num_months': len(delivery_months),
            'sample_roll_dates': ' | '.join(sample_dates),
            'data_start': non['Date'].min(),
            'data_end': non['Date'].max(),
        })
    
    return results


def main():
    print("=" * 70)
    print("CLC Contract Months Extraction (from RAD/NON ratio jumps)")
    print("=" * 70)
    
    results = analyze_all_contracts()
    df = pd.DataFrame(results)
    
    # Sort by rule_type, then symbol
    df = df.sort_values(['rule_type', 'symbol']).reset_index(drop=True)
    
    # Save full results
    output_file = DATA_DIR / 'contract_months_extracted.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Print summary
    print(f"\nTotal contracts analyzed: {len(df)}")
    print(f"With rolls detected: {len(df[df['roll_count'] > 0])}")
    
    # Group by rule type
    print("\n--- By Rule Type ---")
    for rule_type in df['rule_type'].unique():
        subset = df[df['rule_type'] == rule_type]
        print(f"\n{rule_type} ({len(subset)} contracts):")
        for _, row in subset.iterrows():
            print(f"  {row['symbol']:4s} → {row['contract_months']:30s} ({row['num_months']} months, {row['roll_count']} rolls)")
    
    # Check for anomalies
    print("\n--- Potential Anomalies ---")
    for _, row in df.iterrows():
        if row['roll_count'] == 0:
            print(f"  {row['symbol']}: No rolls detected!")
        elif row['rule_type'] == 'UNKNOWN':
            print(f"  {row['symbol']}: No roll rule in config")
    
    print("\n" + "=" * 70)
    print("Done. Review the output and compare with external data.")
    print("=" * 70)


if __name__ == '__main__':
    main()