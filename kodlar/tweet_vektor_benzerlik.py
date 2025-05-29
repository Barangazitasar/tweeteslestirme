import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Modeli ve veriyi yükle
model = Word2Vec.load("word2vec_model.model")
df = pd.read_csv("temizlenmis_tweetler_stop.csv")
df["TemizTweet_Stop"] = df["TemizTweet_Stop"].fillna("")

# Ortalama vektör hesaplayan fonksiyon
def tweet_vektor(tweet):
    kelimeler = tweet.lower().split()
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    if len(vektorler) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vektorler, axis=0)

# Tüm tweetler için vektörleri hesapla
vektorler = np.array([tweet_vektor(tweet) for tweet in df["TemizTweet_Stop"]])

# Örnek: İlk tweete en benzer 5 tweeti bul
index = 0
benzerlikler = cosine_similarity([vektorler[index]], vektorler)[0]
benzer_siralama = benzerlikler.argsort()[::-1][1:6]  # kendisi hariç ilk 5

print(f"🔍 Seçilen tweet:\n{df.iloc[index]['TemizTweet_Stop']}\n")
print("🧠 En benzer 5 tweet:")
for i in benzer_siralama:
    print(f"- {df.iloc[i]['TemizTweet_Stop']} (Benzerlik: {benzerlikler[i]:.2f})")
