import pandas as pd

# Veriyi yükle
df = pd.read_csv("all_annotated.tsv", sep="\t")

# Sadece Türkiye'den olanları filtrele (Country sütununa göre)
turkce_df = df[df["Country"] == "TR"]

# Filtrelenmiş veriyi kaydet
turkce_df.to_csv("turkce_tweetler.csv", index=False)

print(f"{len(turkce_df)} adet Türkçe tweet başarıyla kaydedildi.")

