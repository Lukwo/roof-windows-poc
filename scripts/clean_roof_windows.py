"""
Run once to create data/roof_windows_uk.parquet from Raw_data_WIN.xlsx
Keeps only rows sold in GB, splits comma-separated sizes,
and extracts both the numeric value and its note from mixed cells.
"""

import re, pathlib, pandas as pd

RAW = pathlib.Path("data/Raw_data_WIN.xlsx")
OUT = pathlib.Path("data/roof_windows_uk.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(RAW)

# 1 · keep only rows whose “Available markets” cell contains GB
df = df[df["Available markets"].fillna("").str.contains(r"\bGB\b", regex=True)]

# 2 · split “Available sizes” into separate rows
df = df.assign(size=df["Available sizes"].str.split(r",\s*")).explode("size")
df["size"] = df["size"].str.strip()

# 3 · make column names computer-friendly
slug = lambda s: re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")
df.columns = [slug(c) for c in df.columns]

# 4 · split cells that look like “1,3 (measured)” into two columns
pattern = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:\((.+)\))?")
num_cols = [c for c in df.columns if any(token in c for token in ("ug", "uw", "g_factor"))]

for c in num_cols:
    m = df[c].astype(str).str.extract(pattern)
    df[f"{c}_num"]  = m[0].str.replace(",", ".").astype(float)
    df[f"{c}_note"] = m[1]
# --- force every object column that LOOKS numeric to float -------------
for col in df.select_dtypes(include=["object"]).columns:
    num = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    if num.notna().sum() >= len(df) * 0.8:    # ≥80 % of cells are numbers
        df[col] = num
    else:
        df[col] = df[col].astype(str)

df.to_parquet(OUT, index=False)
print("✅ Clean file created →", OUT)
