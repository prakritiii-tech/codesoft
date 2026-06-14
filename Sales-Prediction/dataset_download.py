import pandas as pd

url = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Advertising.csv"

df = pd.read_csv(url)

df.to_csv("Advertising.csv", index=False)

print("Dataset Downloaded Successfully!")
print(df.head())