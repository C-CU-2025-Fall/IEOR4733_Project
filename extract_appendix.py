#!/usr/bin/env python3
"""Extract appendix from paper PDF to check futures contracts."""

import subprocess
import sys

# Try using PyPDF2 or similar
try:
    import PyPDF2
    
    with open('/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/Deep_Reinforcement_learning_trading.pdf', 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        print(f"Total pages: {len(reader.pages)}")
        
        # Check last few pages for appendix
        for i in range(len(reader.pages) - 1, max(0, len(reader.pages) - 5), -1):
            page = reader.pages[i]
            text = page.extract_text()
            if 'appendix' in text.lower() or 'contract' in text.lower() or 'futures' in text.lower():
                print(f"\n=== Page {i+1} ===")
                print(text[:3000])
                
except ImportError:
    print("PyPDF2 not installed, trying pdfminer...")
    
try:
    from pdfminer.high_level import extract_text
    
    text = extract_text('/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/Deep_Reinforcement_learning_trading.pdf')
    
    # Find appendix
    idx = text.lower().find('appendix')
    if idx >= 0:
        print("=== APPENDIX FOUND ===")
        print(text[idx:idx+5000])
    else:
        # Look for table of futures
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'futures' in line.lower() and ('table' in line.lower() or 'contract' in line.lower()):
                print(f"Line {i}: {line}")
                # Print surrounding context
                for j in range(max(0, i-5), min(len(lines), i+20)):
                    print(lines[j])
                print("---")
                
except ImportError:
    print("pdfminer not installed either")
    print("Please install: pip install PyPDF2 or pdfminer.six")