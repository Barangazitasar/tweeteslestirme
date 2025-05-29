import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Türkçe stopwords listesi
turkce_stopwords = set(stopwords.words('turkish'))

# Tokenizer'ı tanımla
tokenizer = TreebankWordTokenizer()

# Veriyi oku
df = pd.read_csv("temizlenmis_tweetler.csv")

# Temizleme fonksiyonu
def temizle_stopwords(metin):
    metin = metin.lower()
    kelimeler = tokenizer.tokenize(metin)
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in turkce_stopwords]
    return " ".join(temiz_kelimeler)

# Stopwords temizliği uygula
df["TemizTweet_Stop"] = df["TemizTweet"].astype(str).apply(temizle_stopwords)

# Yeni dosyaya kaydet
df.to_csv("temizlenmis_tweetler_stop.csv", index=False)

print("Stopwords temizliği tamamlandı ve dosya kaydedildi.")
