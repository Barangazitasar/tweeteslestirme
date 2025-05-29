import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# 🔹 Basit tokenizer (nltk yerine)
def tokenize(text):
    return text.lower().split()

# 🔹 Ortalama vektör hesabı
def ortalama_vektor(tweet, model):
    if not tweet:
        return np.zeros(model.vector_size)
    vektorler = [model.wv[word] for word in tweet if word in model.wv]
    if vektorler:
        return np.mean(vektorler, axis=0)
    else:
        return np.zeros(model.vector_size)

# 🔹 Veriyi oku
df = pd.read_csv("temizlenmis_tweetler_stop.csv")
df["TemizTweet_Stop"] = df["TemizTweet_Stop"].fillna("")

# 🔹 Tweetleri tokenize et
tweet_tokens = [tokenize(tweet) for tweet in df["TemizTweet_Stop"]]

# 🔹 Eğitilmiş Word2Vec modelini yükle
model = Word2Vec.load("word2vec_model.model")

# 🔹 Tüm tweetlerin ortalama vektörlerini hesapla
vektorler = np.array([ortalama_vektor(tweet, model) for tweet in tweet_tokens])

# 🔹 Belirli bir tweet'e göre en benzerleri bul
def benzer_tweetleri_getir(index, k=5):
    secilen = vektorler[index]
    benzerlikler = np.dot(vektorler, secilen) / (
        np.linalg.norm(vektorler, axis=1) * np.linalg.norm(secilen) + 1e-10
    )
    en_benzer_indeksler = np.argsort(benzerlikler)[::-1][1:k+1]
    return en_benzer_indeksler

# 🔹 Örnek: 10. tweet'e en benzer 5 tweet
index = 10
print(f"📌 Seçilen tweet: {df['TemizTweet_Stop'][index]}")
print("\n🔁 Benzer tweetler:")
for i in benzer_tweetleri_getir(index):
    print(f"- {df['TemizTweet_Stop'][i]}")
