import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Veriyi yükle
df = pd.read_csv("temizlenmis_tweetler_stop.csv")
tweets = df["TemizTweet_Stop"].fillna("").tolist()

# --- TF-IDF vektörleri oluştur ---
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

# --- Word2Vec modelini yükle ---
model = Word2Vec.load("word2vec_model.model")

# Tweet başına ortalama Word2Vec vektörü hesapla
def ortalama_vektor(tweet, model):
    kelimeler = tweet.split()
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    if vektorler:
        return np.mean(vektorler, axis=0)
    else:
        return np.zeros(model.vector_size)

w2v_vektorler = np.array([ortalama_vektor(tw, model) for tw in tweets])

# --- Örnek bir tweet seç ---
secilen_index = 0
print("🔴 TF-IDF & Word2Vec karşılaştırması")
print("🔸 Seçilen tweet:", tweets[secilen_index])

# --- TF-IDF benzerlikleri ---
tfidf_scores = cosine_similarity(tfidf_matrix[secilen_index], tfidf_matrix)[0]
tfidf_indices = tfidf_scores.argsort()[-6:-1][::-1]  # En benzer 5 (ilk tweet hariç)

# --- Word2Vec benzerlikleri ---
w2v_scores = cosine_similarity([w2v_vektorler[secilen_index]], w2v_vektorler)[0]
w2v_indices = w2v_scores.argsort()[-6:-1][::-1]

# --- Sonuçları yazdır ---
print("\n📌 TF-IDF ile en benzer tweetler:")
for i in tfidf_indices:
    print("-", tweets[i])

print("\n📌 Word2Vec ile en benzer tweetler:")
for i in w2v_indices:
    print("-", tweets[i])
