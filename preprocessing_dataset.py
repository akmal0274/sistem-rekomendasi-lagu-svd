import pandas as pd

# 1. Baca file CSV
df = pd.read_csv("data/songs_normalize.csv")  # ganti dengan nama file kamu

# 2. Tampilkan jumlah awal
print("Jumlah sebelum hapus duplikat:", len(df))

# 3. Hapus duplikat (berdasarkan semua kolom secara default)
df_unique = df.drop_duplicates(subset=["song", "artist"])

# 4. Tampilkan jumlah setelahnya
print("Jumlah setelah hapus duplikat:", len(df_unique))

# 5. Simpan kembali (opsional)
df_unique.to_csv("data/songs_normalize_clean.csv", index=False)
