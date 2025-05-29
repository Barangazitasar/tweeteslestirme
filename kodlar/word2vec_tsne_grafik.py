import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

model = Word2Vec.load("word2vec_model.model")
kelimeler = list(model.wv.index_to_key[:50])
vektorler = np.array([model.wv[kelime] for kelime in kelimeler]) 



# TSNE ile 2 boyuta indir
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vektorler_2d = tsne.fit_transform(vektorler)

# Grafik çiz
plt.figure(figsize=(12, 8))
for i, kelime in enumerate(kelimeler):
    x, y = vektorler_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.5, y + 0.5, kelime, fontsize=9)

plt.title("Word2Vec Kelime Vektörleri (TSNE ile 2D)")
plt.xlabel("Boyut 1")
plt.ylabel("Boyut 2")
plt.grid(True)
plt.show()
