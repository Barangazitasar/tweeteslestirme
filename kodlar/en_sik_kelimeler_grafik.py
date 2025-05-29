import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Temizlenmiş veriyi yükle
df = pd.read_csv("temizlenmis_tweetler_stop.csv")

# Boş olmayan tweetleri al
tweetler = df["TemizTweet_Stop"].dropna().tolist()

# Kelimeleri tek listede birleştir
kelimeler = []
for tweet in tweetler:
    kelimeler.extend(tweet.split())

# En sık geçen 20 kelimeyi say
kelime_sayilari = Counter(kelimeler).most_common(20)

# Bar grafiğini çiz
kelimeler, sayilar = zip(*kelime_sayilari)
plt.figure(figsize=(12, 6))
plt.bar(kelimeler, sayilar, color="skyblue")
plt.xticks(rotation=45)
plt.title("En Sık Geçen 20 Kelime")
plt.xlabel("Kelimeler")
plt.ylabel("Frekans")
plt.tight_layout()
plt.show()
