import pandas as pd
from gensim.models import Word2Vec

# Veriyi oku
df = pd.read_csv("temizlenmis_tweetler_stop.csv")

# Boş değerleri temizle
df["TemizTweet_Stop"] = df["TemizTweet_Stop"].fillna("")

# Tokenize et (sadece boşluklara göre ayır)
tweets = [tweet.lower().split() for tweet in df["TemizTweet_Stop"]]

# Word2Vec modelini eğit
model = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=2, workers=4)

# Modeli kaydet
model.save("word2vec_model.model")

print("✅ Word2Vec modeli başarıyla eğitildi ve kaydedildi.")
