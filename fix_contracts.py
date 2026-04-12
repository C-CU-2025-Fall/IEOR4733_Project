#!/usr/bin/env python3
"""
fix_contracts.py — Rebuild FIXED.CSV for contracts excluded due to roll/data issues.

Method (Scheme B — ratio splice):
  Uses the _NON.CSV (unadjusted continuous) as base series.
  Wherever the absolute daily return exceeds `threshold`, a correction factor is
  applied so that the adjusted close matches the previous day's close exactly at
  that point. The factor is then kept for all subsequent rows (forward propagation).

  This removes mechanical price discontinuities caused by bad ratio-adjustment
  without altering the inter-day returns on normal trading days.

Usage:
    python fix_contracts.py                          # threshold=10%, default dirs
    python fix_contracts.py --threshold 0.05         # tighter 5% threshold
    python fix_contracts.py --threshold 0.15         # looser 15% threshold
    python fix_contracts.py --source-dir path/to/CLCDATA --target-dir data/CLC
"""
import argparse
import csv
from pathlib import Path

FIXED_TICKERS = {
    # Commodity (25)
    'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',
    'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZU',
    'ZO', 'ZP', 'ZR', 'ZT', 'ZW', 'ZZ',
    # Equity Index (11)
    'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM',
    # Fixed Income (5)
    'DT', 'FB', 'TY', 'UB', 'US',
    # Forex (9)
    'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN',
}

DEFAULT_SOURCE_DIR = Path(__file__).parent.parent / 'data-check' / 'CLCDATA'
DEFAULT_TARGET_DIR = Path(__file__).parent / 'data' / 'CLC'


def fmt_price(value: float) -> str:
    text = f'{value:.10f}'.rstrip('0').rstrip('.')
    return text if text else '0'


def fix_one(ticker: str, source_dir: Path, target_dir: Path, threshold: float) -> dict:
    src = source_dir / f'{ticker}_NON.CSV'
    dst = target_dir / f'{ticker}_FIXED.CSV'

    with src.open(newline='') as f:
        rows = list(csv.reader(f))

    if not rows:
        raise RuntimeError(f'{ticker}: source file is empty')

    factor = 1.0
    prev_close = None
    adjustments = 0
    output_rows = []

    for row in rows:
        date_str, opn, high, low, close, vol, oi = row
        raw = [float(opn), float(high), float(low), float(close)]

        if prev_close is not None and raw[3] > 0:
            projected_close = raw[3] * factor
            jump = abs(projected_close / prev_close - 1.0)
            if jump > threshold:
                factor = prev_close / raw[3]
                adjustments += 1

        adj = [p * factor for p in raw]
        prev_close = adj[3]
        output_rows.append([date_str,
                             fmt_price(adj[0]),
                             fmt_price(adj[1]),
                             fmt_price(adj[2]),
                             fmt_price(adj[3]),
                             vol, oi])

    target_dir.mkdir(parents=True, exist_ok=True)
    with dst.open('w', newline='') as f:
        csv.writer(f).writerows(output_rows)

    return {'ticker': ticker, 'total_rows': len(output_rows), 'adjustments': adjustments, 'dst': str(dst)}


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild FIXED.CSV for five problematic contracts using NON data + ratio splicing')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='Absolute daily return threshold for splice (default: 0.10 = 10%%)')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR,
                        help=f'Directory containing *_NON.CSV files (default: {DEFAULT_SOURCE_DIR})')
    parser.add_argument('--target-dir', type=Path, default=DEFAULT_TARGET_DIR,
                        help=f'Directory to write *_FIXED.CSV files (default: {DEFAULT_TARGET_DIR})')
    args = parser.parse_args()

    print(f'\nfix_contracts.py — threshold={args.threshold:.0%}')
    print(f'  source : {args.source_dir}')
    print(f'  target : {args.target_dir}')
    print()

    for ticker in sorted(FIXED_TICKERS):
        result = fix_one(ticker, args.source_dir, args.target_dir, args.threshold)
        print(f'  {ticker}  rows={result["total_rows"]}  splice_points={result["adjustments"]}  \u2192 {result["dst"]}')

    print(f'\nDone. {len(FIXED_TICKERS)} contracts written. Loader reads *_FIXED.CSV for all of them.')


if __name__ == '__main__':
    main()
