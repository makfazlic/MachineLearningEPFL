import pandas as pd

df = pd.read_csv("svm.csv")

for col in df.columns:
    if col != "Unnamed: 0":
        # Print max
        print(f"Max for {col}: {df[col].max()}")
        for i, row in enumerate(df[col]):
            a = float(row.split(" ")[0].replace(",", "").replace("(", ""))
            if a > 96:
                print("Found, ", a, "for", col, "and", i)

