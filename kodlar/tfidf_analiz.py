import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Temizlenmiş veriyi yükle
df = pd.read_csv("temizlenmis_tweetler_stop.csv")

# NaN değerleri boş string ile değiştir
df["TemizTweet_Stop"] = df["TemizTweet_Stop"].fillna("")

# TF-IDF vektörizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["TemizTweet_Stop"])

# Terimlerin isimleri
kelimeler = vectorizer.get_feature_names_out()

# İlk 10 tweet için TF-IDF skorları
df_tfidf = pd.DataFrame(tfidf_matrix[:10].toarray(), columns=kelimeler)

# İlk satırı göster
print(df_tfidf.head())

# CSV’ye kaydet (isteğe bağlı)
df_tfidf.to_csv("tfidf_sonuclari.csv", index=False)
