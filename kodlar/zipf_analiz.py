import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Veriyi yükle
df = pd.read_csv("temizlenmis_tweetler_stop.csv")

# Tüm kelimeleri birleştir
tum_metin = " ".join(df["TemizTweet_Stop"].dropna().astype(str))

# Kelimelere ayır
kelimeler = tum_metin.split()

# Frekansları hesapla
frekanslar = Counter(kelimeler)
en_yaygin = frekanslar.most_common()

# Sıra (rank) ve frekans değerleri
rank = np.arange(1, len(en_yaygin) + 1)
frequency = np.array([f for _, f in en_yaygin])

# Log-log grafik
plt.figure(figsize=(10, 6))
plt.plot(np.log(rank), np.log(frequency))
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.title("Zipf Yasası - Tweet Verisi")
plt.grid(True)
plt.show()
