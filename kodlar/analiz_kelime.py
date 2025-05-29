import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv("temizlenmis_tweetler.csv")

# Tüm tweetleri birleştir
tum_metin = " ".join(df["TemizTweet"].dropna())

# Kelimeleri ayır
kelimeler = tum_metin.split()

# Frekans hesapla
frekanslar = Counter(kelimeler)

# En sık geçen 20 kelimeyi al
en_sik = frekanslar.most_common(20)

# Görselleştir
kelimeler, sayilar = zip(*en_sik)
plt.figure(figsize=(10,5))
plt.bar(kelimeler, sayilar)
plt.xticks(rotation=45)
plt.title("En Sık Geçen 20 Kelime")
plt.show()
