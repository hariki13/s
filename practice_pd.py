import pandas as pd

df = pd.read_csv("csv/data/starbucks_ny_reserve_roastery.csv")

# print(df.loc["11:00"])
# selection bya row index
# print(df.iloc[660:670:500])   

print(df)

# selection by column name
# ror = input("what ror roast trigger:")

# try:
    # print(df.loc[ror])
# except KeyError:
    # print(f"ROR {ror} not found in the data.")