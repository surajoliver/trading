import pandas as pd
from pathlib import Path
import yfinance as yf
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "Stock Data"
INDEX_DIR = ROOT / "data" / "raw" / "Indices"

def main():
    print('Reading stock indices from CSV...')
    for index in ['nifty50', 'niftynext50', 'nifty500', 'niftymidcap150', 'niftysmallcap250']:
        indices_df = pd.read_csv(INDEX_DIR / f'{index}.csv', index_col=0)
        stocks = indices_df['Yahoo'].unique().tolist()
        print(f"Index: {index} | Found stocks: {len(stocks)}")

        data = yf.download(
            stocks, 
            start='2010-01-01', 
            end='2026-12-31',
            auto_adjust=True,
            progress=False
        )
        data.to_csv(DATA_DIR / f'{index}_data.csv')

        data.to_pickle(DATA_DIR / f'{index}_data.pkl')

    

if __name__ == "__main__":
    main()