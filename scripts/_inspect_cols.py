import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
print(f"Rows: {len(df)}  |  Cols: {len(df.columns)}")
for c in df.columns[:30]:
    print(c)
if len(df.columns) > 30:
    print(f"... and {len(df.columns)-30} more")
