import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# Modeli yükle
model = Word2Vec.load("word2vec_model.model")

# En sık geçen 50 kelimeyi al
kelimeler = list(model.wv.index_to_key[:50])
vektorler = [model.wv[kelime] for kelime in kelimeler]

# PCA ile 2 boyuta indir
pca = PCA(n_components=2)
vektorler_2d = pca.fit_transform(vektorler)

# Grafik çiz
plt.figure(figsize=(12, 8))
for i, kelime in enumerate(kelimeler):
    x, y = vektorler_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, kelime, fontsize=9)

plt.title("Word2Vec Kelime Vektörleri (PCA ile 2D)")
plt.xlabel("Boyut 1")
plt.ylabel("Boyut 2")
plt.grid(True)
plt.show()
