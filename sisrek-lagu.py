import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import pandas as pd
import os

# ======== Load Users CSV ========
def load_users(file_path="data/users.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=["username", "name", "password"])

# ======== Save User to CSV ========
def save_user(username, name, hashed_password, file_path="data/users.csv"):
    new_user = pd.DataFrame([[username, name, hashed_password]], columns=["username", "name", "password"])
    users_df = load_users(file_path)
    
    if username in users_df["username"].values:
        return False  # username sudah ada
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(file_path, index=False)
    return True

# ======== Auth System ========
def build_credentials(df_users):
    return {
        "usernames": {
            row["username"]: {
                "name": row["name"],
                "password": row["password"]
            } for _, row in df_users.iterrows()
        }
    }

# ======== UI Utama ========
def main():
    st.title("Sistem Login & Register - Rekomendasi Lagu")

    menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

    if menu == "Login":
        df_users = load_users()
        credentials = build_credentials(df_users)

        authenticator = stauth.Authenticate(
            credentials,
            cookie_name="rekomendasi_lagu",
            key="random_key",
            cookie_expiry_days=1
        )

        authenticator.login(location="main")

        if st.session_state.get('authentication_status'):
            st.success(f"Halo, {st.session_state.get('name')} 👋")
            st.session_state["user_id"] = st.session_state.get('username')
            st.info(f"User aktif: {st.session_state.get('username')}")

            st.subheader("🎵 Pilih Lagu Favorit Anda")

                # Load dataset lagu
                df_songs = pd.read_csv("data/songs_normalize.csv")
                user_likes_path = "data/user_likes.csv"
                
                # Cek apakah user sudah pernah memilih lagu
                user_likes = pd.read_csv(user_likes_path) if os.path.exists(user_likes_path) else pd.DataFrame(columns=["user_id", "song"])
                existing_likes = user_likes[user_likes["user_id"] == st.session_state["user_id"]]

                if existing_likes.empty:
                    st.subheader("🎵 Pilih Lagu Favorit Anda")
                    all_songs = df_songs['song'] + " - " + df_songs['artist']
                    
                    selected_songs = st.multiselect("Pilih lagu yang Anda sukai:", all_songs)

                    if st.button("Simpan Lagu Favorit"):
                        if selected_songs:
                            likes = pd.DataFrame({
                                "user_id": st.session_state["user_id"],
                                "song": [s.split(" - ")[0] for s in selected_songs]
                            })

                            updated_likes = pd.concat([user_likes, likes], ignore_index=True)
                            updated_likes.to_csv(user_likes_path, index=False)
                            st.success("Lagu favorit berhasil disimpan! 🎉")
                        else:
                            st.warning("Pilih setidaknya satu lagu.")
                else:
                    st.success("Selamat datang kembali!")
                    st.write("🎧 Lagu favorit Anda sebelumnya:")
                    st.dataframe(existing_likes[["song"]])

                    # Placeholder untuk tombol rekomendasi nanti
                    if st.button("🎵 Tampilkan Rekomendasi Lagu"):
                        st.info("Fitur rekomendasi akan ditampilkan di sini...")


        elif st.session_state.get('authentication_status') is False:
            st.error("Username atau password salah.")
        elif st.session_state.get('authentication_status') is None:
            st.warning("Masukkan username dan password untuk login.")



    elif menu == "Register":
        st.subheader("Daftar Akun Baru")

        with st.form("register_form", clear_on_submit=True):
            username = st.text_input("Username")
            name = st.text_input("Nama Lengkap")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Konfirmasi Password", type="password")
            submit = st.form_submit_button("Daftar")

        if submit:
            if not username or not name or not password or not confirm_password:
                st.warning("Harap lengkapi semua kolom.")
            elif password != confirm_password:
                st.error("Password dan konfirmasi tidak cocok.")
            else:
                hashed_pw = Hasher([password]).generate()[0]
                success = save_user(username, name, hashed_pw)

                if success:
                    st.success("Berhasil mendaftar! Silakan login dari menu.")
                else:
                    st.error("Username sudah terdaftar. Gunakan username lain.")

if __name__ == "__main__":
    main()
