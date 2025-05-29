import pandas as pd
import re

# CSV dosyasını oku
df = pd.read_csv("turkce_tweetler.csv")

# Sütun adlarını kontrol et
print("Mevcut sütunlar:", df.columns.tolist())

# Tweet verisinin bulunduğu sütun adını belirt (çıktıya göre düzenle!)
sütun_adi = "Tweet"  # Örn: "Tweet Text" veya "Tweet", CSV dosyana göre

# Temizleme fonksiyonu
def temizle(metin):
    if pd.isna(metin):
        return ""
    metin = metin.lower()
    metin = re.sub(r"http\S+", "", metin)     # link sil
    metin = re.sub(r"@\w+", "", metin)        # mention sil
    metin = re.sub(r"#\w+", "", metin)        # hashtag sil
    metin = re.sub(r"\d+", "", metin)         # sayı sil
    metin = re.sub(r"[^\w\sçğıöüşİÇĞÖÜŞ]", "", metin)  # noktalama sil (Türkçeye dikkat!)
    metin = re.sub(r"\s+", " ", metin)        # fazla boşluk sil
    return metin.strip()

# Temiz sütun oluştur
df["TemizTweet"] = df[sütun_adi].astype(str).apply(temizle)

# Sonucu kaydet
df.to_csv("temizlenmis_tweetler.csv", index=False)

print("Temizleme tamamlandı ve dosya kaydedildi.")
