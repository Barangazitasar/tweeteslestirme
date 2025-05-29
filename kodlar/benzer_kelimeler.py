from gensim.models import Word2Vec

# Eğittiğimiz modeli yükle
model = Word2Vec.load("word2vec_model.model")

# Örnek olarak 'türk' kelimesine en benzeyen 10 kelimeyi getir
try:
    benzerler = model.wv.most_similar("türk", topn=10)

    print("🔍 'türk' kelimesine en yakın kelimeler:")
    for kelime, skor in benzerler:
        print(f"{kelime} -> {skor:.2f}")
except KeyError:
    print("Hata: 'seçim' kelimesi modelde bulunamadı. Lütfen başka bir kelime deneyin.")
