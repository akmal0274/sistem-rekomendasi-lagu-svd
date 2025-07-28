# import pandas as pd

# # 1. Baca file CSV
# df = pd.read_csv("data/songs_normalize.csv")

# # 2. Tampilkan jumlah awal
# print("Jumlah sebelum hapus duplikat:", len(df))

# # 3. Hapus duplikat berdasarkan kombinasi song dan artist
# df_unique = df.drop_duplicates(subset=["song", "artist"]).reset_index(drop=True)

# # 4. Tambahkan kolom song_id sebagai angka unik
# df_unique["song_id"] = df_unique.index  # 0, 1, 2, ...

# # 5. Tampilkan jumlah setelahnya
# print("Jumlah setelah hapus duplikat:", len(df_unique))

# # 6. Simpan ke file baru
# df_unique.to_csv("data/songs_normalize_clean.csv", index=False)

import pandas as pd

# 1. Load file songs yang sudah dibersihkan (sudah punya song_id)
songs_df = pd.read_csv("data/songs_normalize_clean.csv")

# 2. Load file user_likes
likes_df = pd.read_csv("data/user_likes.csv")

# 3. Gabungkan berdasarkan 'song' dan 'artist' agar dapat song_id
likes_with_id = pd.merge(likes_df, songs_df[['song', 'artist', 'song_id']],
                         on=['song', 'artist'], how='left')

# 4. Cek jika ada yang gagal dipasangkan (NaN)
missing = likes_with_id[likes_with_id['song_id'].isna()]
if not missing.empty:
    print("❗ Ada lagu dari user_likes yang tidak ditemukan di songs_normalize_clean.csv:")
    print(missing[['user_id', 'song', 'artist']])
else:
    print("✅ Semua lagu berhasil dipasangkan dengan song_id.")

# 5. Simpan hasil akhir (gunakan hanya user_id dan song_id)
likes_with_id_clean = likes_with_id[['user_id', 'song_id']]
likes_with_id_clean.to_csv("data/user_likes_with_id.csv", index=False)
print("📁 Disimpan ke data/user_likes_with_id.csv")
