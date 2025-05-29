import pandas as pd

# Veri dosyasını oku
df = pd.read_csv("all_annotated.tsv", sep="\t")

# İlk 5 satırı yazdır
print("Sütunlar:", df.columns)
print(df.head())
