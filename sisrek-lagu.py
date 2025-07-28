import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import pandas as pd
import os
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_authenticator():
    """Get authenticator instance"""
    df_users = load_users()
    if df_users.empty:
        return None
        
    credentials = build_credentials(df_users)
    return stauth.Authenticate(
        credentials,
        cookie_name="rekomendasi_lagu",
        key="random_key",
        cookie_expiry_days=0
    )

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
    os.makedirs("data", exist_ok=True)
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

def content_based_recommendation(user_id, songs_df, user_likes_df, top_n=10):
    print(f"CBF: Processing for user {user_id}")
    
    # Ambil song_id yang disukai user
    liked_song_ids = user_likes_df[user_likes_df['user_id'] == user_id]['song_id'].tolist()
    print(f"CBF: User liked {len(liked_song_ids)} songs: {liked_song_ids[:3]}...")

    if not liked_song_ids:
        print("CBF: No liked songs found")
        return pd.DataFrame()

    songs_copy = songs_df.copy()
    liked_features = songs_copy[songs_copy['song_id'].isin(liked_song_ids)]

    if liked_features.empty:
        print("CBF: No matching features found for liked songs")
        return pd.DataFrame()

    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo']
    
    feature_cols = [col for col in feature_cols if col in songs_copy.columns]
    print(f"CBF: Using features: {feature_cols}")

    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(songs_copy[feature_cols])
    liked_features_scaled = scaler.transform(liked_features[feature_cols])

    # Profil user: rata-rata fitur lagu yang disukai
    user_profile = liked_features_scaled.mean(axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_profile, all_features_scaled)[0]

    songs_copy['similarity'] = similarities

    # Hindari lagu yang sudah disukai
    recommendations = songs_copy[~songs_copy['song_id'].isin(liked_song_ids)].copy()
    recommendations = recommendations.sort_values(by='similarity', ascending=False)

    # Ambil top-N rekomendasi
    top_recs = recommendations[['song_id', 'song', 'artist', 'similarity']].head(top_n)
    top_recs = top_recs.rename(columns={"similarity": "estimated_rating"})

    print(f"CBF: Generated {len(top_recs)} recommendations")
    print(f"CBF: Score range: {top_recs['estimated_rating'].min():.4f} to {top_recs['estimated_rating'].max():.4f}")

    return top_recs



def collaborative_filtering_recommendation(user_id, df_songs, user_likes, top_n=10):
    print(f"CF: Processing for user {user_id}")
    
    unique_users = user_likes['user_id'].nunique()
    unique_songs = user_likes['song_id'].nunique()
    print(f"CF: Dataset has {unique_users} users and {unique_songs} songs")
    
    if unique_users < 2 or unique_songs < 2:
        print("CF: Not enough data for collaborative filtering")
        return pd.DataFrame()

    df_ratings = user_likes.copy()
    df_ratings["rating"] = 1.0

    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(df_ratings))
    df_ratings["rating"] += noise
    df_ratings["rating"] = df_ratings["rating"].clip(0.1, 1.0)

    # Gunakan song_id di Surprise
    reader = Reader(rating_scale=(0.1, 1.0))
    data = Dataset.load_from_df(df_ratings[["user_id", "song_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(random_state=42, n_factors=10, n_epochs=20)
    model.fit(trainset)

    # 💡 Evaluasi MAE dan RMSE
    predictions = model.test(testset)
    mae = accuracy.mae(predictions, verbose=False)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"CF: Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    user_liked_song_ids = set(user_likes[user_likes["user_id"] == user_id]["song_id"])
    all_song_ids = set(df_songs["song_id"])
    unseen_song_ids = list(all_song_ids - user_liked_song_ids)

    print(f"CF: Predicting for {len(unseen_song_ids)} unseen songs")

    recommendations = []
    for song_id in unseen_song_ids:
        pred = model.predict(user_id, song_id)
        recommendations.append((song_id, pred.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_predictions = recommendations[:top_n]

    rec_df = pd.DataFrame(top_predictions, columns=["song_id", "estimated_rating"])
    
    # Gabungkan dengan info lagu
    final_recs = pd.merge(rec_df, df_songs[["song_id", "song", "artist"]], on="song_id", how="left")

    print(f"CF: Generated {len(final_recs)} recommendations")
    if not final_recs.empty:
        print(f"CF: Score range: {final_recs['estimated_rating'].min():.4f} to {final_recs['estimated_rating'].max():.4f}")

    # Tambahkan evaluasi ke dalam output (optional)
    final_recs["MAE"] = mae
    final_recs["RMSE"] = rmse

    return final_recs


def hybrid_filtering(user_id, df_songs, user_likes, top_n=10):
    print(f"HYBRID: Starting hybrid recommendation for user {user_id}")
    
    # Get recommendations from both methods
    cbf_recs = content_based_recommendation(user_id, df_songs, user_likes, top_n * 2)
    cf_recs = collaborative_filtering_recommendation(user_id, df_songs, user_likes, top_n * 2)
    
    print(f"HYBRID: CBF returned {len(cbf_recs)} recommendations")
    print(f"HYBRID: CF returned {len(cf_recs)} recommendations")

    # Ensure column consistency
    if cbf_recs.empty and cf_recs.empty:
        print("HYBRID: Both methods failed")
        return pd.DataFrame(columns=['song_id', 'song', 'artist', 'hybrid_score', 'method'])

    def fallback_result(df, method_label):
        df = df.head(top_n).copy()
        if 'estimated_rating' in df.columns:
            df['hybrid_score'] = df['estimated_rating']
        else:
            df['hybrid_score'] = 0.5
        df['method'] = method_label
        return df[['song_id', 'song', 'artist', 'hybrid_score', 'method']].reset_index(drop=True)

    if cbf_recs.empty:
        print("HYBRID: Using only CF recommendations")
        return fallback_result(cf_recs, 'cf_only')
    
    if cf_recs.empty:
        print("HYBRID: Using only CBF recommendations")
        return fallback_result(cbf_recs, 'cbf_only')

    # Normalize scores
    def normalize_scores(df, score_col):
        df = df.copy()
        if df.empty or df[score_col].std() == 0:
            df[score_col + '_normalized'] = 0.5
        else:
            min_score = df[score_col].min()
            max_score = df[score_col].max()
            if max_score == min_score:
                df[score_col + '_normalized'] = 0.5
            else:
                df[score_col + '_normalized'] = (df[score_col] - min_score) / (max_score - min_score)
        return df

    cbf_recs = normalize_scores(cbf_recs, 'estimated_rating')
    cf_recs = normalize_scores(cf_recs, 'estimated_rating')

    # Convert to dictionary for lookup
    cbf_scores = dict(zip(cbf_recs["song_id"], cbf_recs["estimated_rating_normalized"]))
    cf_scores = dict(zip(cf_recs["song_id"], cf_recs["estimated_rating_normalized"]))

    # Combine all song_ids
    all_song_ids = set(cbf_scores.keys()).union(cf_scores.keys())
    print(f"HYBRID: Combining {len(all_song_ids)} unique song_ids")

    # Weighted average
    cbf_weight, cf_weight = 0.6, 0.4
    hybrid_scores = []
    for song_id in all_song_ids:
        cbf_score = cbf_scores.get(song_id, 0)
        cf_score = cf_scores.get(song_id, 0)

        if song_id in cbf_scores and song_id in cf_scores:
            score = (cbf_weight * cbf_score) + (cf_weight * cf_score)
            method = "both"
        elif song_id in cbf_scores:
            score = cbf_score * 0.8
            method = "cbf_only"
        else:
            score = cf_score * 0.8
            method = "cf_only"

        hybrid_scores.append((song_id, score, method))

    # Sort and prepare result
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_hybrid = hybrid_scores[:top_n]

    # Merge with song info
    top_hybrid_df = pd.DataFrame(top_hybrid, columns=["song_id", "hybrid_score", "method"])
    merged = pd.merge(top_hybrid_df, df_songs[['song_id', 'song', 'artist']], on='song_id', how='left')

    result = merged[['song_id', 'song', 'artist', 'hybrid_score', 'method']].reset_index(drop=True)

    print(f"HYBRID: Final recommendations:")
    for i, row in result.iterrows():
        print(f"  {row['song_id']} - {row['song']} - {row['artist']}: {row['hybrid_score']:.4f} ({row['method']})")

    return result


def initialize_session_state():
    """Initialize session state variables"""
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None
    if "name" not in st.session_state:
        st.session_state["name"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "logout_clicked" not in st.session_state:
        st.session_state["logout_clicked"] = False

def handle_login():
    """Handle login functionality for streamlit_authenticator 0.4.2"""
    st.subheader("🔐 Login")
    
    authenticator = st.session_state.get("authenticator")
    if authenticator is None:
        st.warning("Belum ada user terdaftar. Silakan register terlebih dahulu.")
        return
    
    # Reset logout flag when trying to login
    st.session_state["logout_clicked"] = False
    
    try:
        authenticator.login(location='main')
    except Exception as e2:
        st.error(f"Fallback login error: {e2}")
        # Last resort: try the old method
        try:
            authenticator.login('main')
        except Exception as e3:
            st.error(f"All login methods failed: {e3}")
            return
    
    # Check session state after login attempt
    if st.session_state.get("authentication_status") == True:
        st.success(f"Halo, {st.session_state.get('name')} 👋")
        st.rerun()
    elif st.session_state.get("authentication_status") == False:
        st.error("Username atau password salah.")
    elif st.session_state.get("authentication_status") is None:
        st.info("Masukkan username dan password untuk login.")

def handle_register():
    """Handle registration functionality"""
    st.subheader("📝 Daftar Akun Baru")
    
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username")
        name = st.text_input("Nama Lengkap")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Konfirmasi Password", type="password")
        submit = st.form_submit_button("Daftar")
        
        if submit:
            if not all([username, name, password, confirm_password]):
                st.warning("Harap lengkapi semua kolom.")
            elif password != confirm_password:
                st.error("Password dan konfirmasi tidak cocok.")
            elif len(password) < 6:
                st.error("Password minimal 6 karakter.")
            else:
                hashed_pw = Hasher().hash(password)
                success = save_user(username, name, hashed_pw)
                if success:
                    st.success("Berhasil mendaftar! Silakan login.")
                    # Refresh authenticator to include new user
                    st.session_state["authenticator"] = get_authenticator()
                else:
                    st.error("Username sudah terdaftar. Pilih username lain.")

def handle_recommendations():
    """Handle music recommendations functionality"""
    # Check if user is logged in and not in logout process
    if (st.session_state.get("authentication_status") != True or 
        st.session_state.get("logout_clicked", False)):
        st.warning("Anda belum login. Silakan login dahulu.")
        return
    
    st.subheader(f"🎵 Rekomendasi Lagu untuk {st.session_state.get('name')}")
    
    # Load songs data
    songs_path = "data/songs_normalize_clean.csv"
    if not os.path.exists(songs_path):
        st.error("File data lagu tidak ditemukan. Pastikan file 'data/songs_normalize.csv' tersedia.")
        return
    
    df_songs = pd.read_csv(songs_path)
    # st.write(f"Dataset contains {len(df_songs)} songs")
    # st.write(f"Dataset columns: {list(df_songs.columns)}")
    
    # Load user likes
    user_likes_path = "data/user_likes.csv"
    if os.path.exists(user_likes_path):
        user_likes = pd.read_csv(user_likes_path)
    else:
        user_likes = pd.DataFrame(columns=["user_id", "song", "artist", "rating"])
    
    # Check if user has selected favorite songs
    existing_likes = user_likes[user_likes["user_id"] == st.session_state["username"]]
    
    if existing_likes.empty:
        st.info("Untuk mendapatkan rekomendasi yang akurat, pilih beberapa lagu favorit Anda terlebih dahulu.")
        
        # Song selection
        all_songs = (df_songs['song'] + " - " + df_songs['artist']).tolist()
        selected_songs = st.multiselect(
            "Pilih lagu yang Anda sukai:",
            all_songs,
            help="Pilih minimal 3 lagu untuk rekomendasi yang lebih baik"
        )
        
        if st.button("💾 Simpan Lagu Favorit", type="primary"):
            if len(selected_songs) < 3:
                st.warning("Pilih minimal 3 lagu untuk rekomendasi yang lebih baik.")
            elif selected_songs:
                # Pisahkan judul dan artis dari input pengguna
                titles = [s.rsplit(" - ", 1)[0] for s in selected_songs]
                artists = [s.rsplit(" - ", 1)[1] for s in selected_songs]

                # Buat DataFrame dari input pengguna
                new_likes_temp = pd.DataFrame({
                    "song": titles,
                    "artist": artists
                })

                # Gabungkan dengan songs_df untuk mendapatkan song_id
                new_likes = pd.merge(new_likes_temp, df_songs[['song', 'artist', 'song_id']],
                                    on=["song", "artist"], how="left")

                # Tambahkan user_id dan rating
                new_likes["user_id"] = st.session_state["username"]
                new_likes["rating"] = 1.0

                # Cek jika ada yang tidak ditemukan
                if new_likes["song_id"].isna().any():
                    st.warning("Beberapa lagu tidak ditemukan dalam database dan tidak disimpan.")
                    st.write(new_likes[new_likes["song_id"].isna()][["song", "artist"]])

                # Hanya simpan yang berhasil ditemukan
                new_likes_clean = new_likes.dropna(subset=["song_id"])[["user_id", "song_id", "song", "artist", "rating"]]

                # Gabungkan dengan data lama
                updated_likes = pd.concat([user_likes, new_likes_clean], ignore_index=True)
                os.makedirs("data", exist_ok=True)
                updated_likes.to_csv(user_likes_path, index=False)

                st.success("Lagu favorit berhasil disimpan!")
                st.balloons()
                st.rerun()
    else:
        # Show user's favorite songs
        st.success("Selamat datang di Sistem Rekomendasi Lagu!")
        
        with st.expander("🎧 Lagu Favorit Anda", expanded=False):
            column_mapping = {
                'song': 'Lagu',
                'artist': 'Artis',
            }
            existing_likes = existing_likes.rename(columns={k: v for k, v in column_mapping.items() if k in existing_likes.columns})
            st.dataframe(
                existing_likes[["Lagu", "Artis"]].reset_index(drop=True),
                use_container_width=True
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🎵 Tampilkan Rekomendasi"):
                with st.spinner("Menganalisis preferensi musik Anda..."):
                    # Enable debug output  
                    hybrid_recs = hybrid_filtering(st.session_state["username"], df_songs, user_likes)
                    
                    if not hybrid_recs.empty:
                        st.subheader("🤝 Rekomendasi Lagu Untuk Anda")
                        
                        
                        # Display with safe column selection
                        available_cols = ['song', 'artist']
                        if 'hybrid_score' in hybrid_recs.columns:
                            available_cols.append('hybrid_score')
                        if 'method' in hybrid_recs.columns:
                            available_cols.append('method')
                        
                        display_df = hybrid_recs[available_cols].copy()
                        
                        # Format score if it exists
                        if 'hybrid_score' in display_df.columns:
                            display_df['hybrid_score'] = display_df['hybrid_score'].round(4)
                        
                        # Rename columns for display
                        column_mapping = {
                            'song': 'Lagu',
                            'artist': 'Artis',
                            'hybrid_score': 'Score',
                            'method': 'Method'
                        }
                        display_df = display_df.rename(columns={k: v for k, v in column_mapping.items() if k in display_df.columns})
                        
                        st.dataframe(
                            display_df.reset_index(drop=True),
                            use_container_width=True
                        )
                        
                        if 'hybrid_score' in hybrid_recs.columns:
                            st.write(f"Score range: {hybrid_recs['hybrid_score'].min():.4f} - {hybrid_recs['hybrid_score'].max():.4f}")
                            st.write(f"Score std: {hybrid_recs['hybrid_score'].std():.4f}")
                    else:
                        st.warning("Tidak cukup data untuk memberikan rekomendasi.")
        
        with col2:
            if st.button("📊 Content-Based Only"):
                with st.spinner("Generating content-based recommendations..."):
                    cbf_recs = content_based_recommendation(st.session_state["username"], df_songs, user_likes)
                    if not cbf_recs.empty:
                        st.subheader("📊 Content-Based Recommendations")
                        column_mapping = {
                            'song': 'Lagu',
                            'artist': 'Artis',
                            'estimated_rating': 'Score',
                        }
                        cbf_recs = cbf_recs.rename(columns={k: v for k, v in column_mapping.items() if k in cbf_recs.columns})
                        st.dataframe(cbf_recs.reset_index(drop=True), use_container_width=True)
                    else:
                        st.warning("Content-based filtering failed.")
        
        with col3:
            if st.button("👥 Collaborative Only"):
                with st.spinner("Generating collaborative filtering recommendations..."):
                    cf_recs = collaborative_filtering_recommendation(st.session_state["username"], df_songs, user_likes)
                    if not cf_recs.empty:
                        st.subheader("👥 Collaborative Filtering Recommendations")
                        column_mapping = {
                            'song': 'Lagu',
                            'artist': 'Artis',
                            'estimated_rating': 'Score',
                        }
                        cf_recs = cf_recs.rename(columns={k: v for k, v in column_mapping.items() if k in cf_recs.columns})
                        st.dataframe(cf_recs.reset_index(drop=True), use_container_width=True)
                    else:
                        st.warning("Collaborative filtering failed because of insufficient data.")
        
        if st.button("🔄 Reset Lagu Favorit"):
            updated_likes = user_likes[user_likes["user_id"] != st.session_state["username"]]
            updated_likes.to_csv(user_likes_path, index=False)
            st.success("Lagu favorit telah direset!")
            st.rerun()

def handle_logout():
    """Handle logout functionality for streamlit_authenticator 0.4.2"""
    st.subheader("👋 Logout")
    
    # Show current status
    st.write(f"Current user: {st.session_state.get('name', 'None')}")
    st.write(f"Authentication status: {st.session_state.get('authentication_status', 'None')}")
    
    if st.button("🚪 Confirm Logout", type="primary"):
        # Set logout flag first
        st.session_state["logout_clicked"] = True
        
        # Try to use authenticator logout if available
        authenticator = st.session_state.get("authenticator")
        if authenticator:
            try:
                # Try different logout method signatures for 0.4.2
                authenticator.logout(button_name='Logout', location='unrendered')
            except Exception as e:
                st.write(f"Authenticator logout failed: {e}")
                # Manual cleanup if authenticator logout fails
                pass
        
        # Manual cleanup - clear all authentication-related session state
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None
        
        # Recreate authenticator to clear any persistent state
        st.session_state["authenticator"] = get_authenticator()
        
        st.success("✅ Berhasil logout!")
        st.info("🔄 Silakan refresh halaman atau pilih menu Login.")
        
        # Force rerun to update the UI
        st.rerun()
    
    # Show logout status
    if st.session_state.get("logout_clicked", False):
        st.info("Anda telah logout. Pilih menu Login untuk masuk kembali.")

def main():
    st.set_page_config(
        page_title="Sistem Rekomendasi Lagu",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Sistem Rekomendasi Lagu")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Force clear authentication if logout was clicked
    if st.session_state.get("logout_clicked", False):
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None
    
    # Initialize authenticator
    if "authenticator" not in st.session_state:
        st.session_state["authenticator"] = get_authenticator()
    
    # Determine if user is logged in (and not in logout process)
    is_logged_in = (st.session_state.get("authentication_status") == True and 
                   not st.session_state.get("logout_clicked", False))
    
    # Sidebar menu
    with st.sidebar:
        st.header("📋 Menu")
        
        if is_logged_in:
            menu_options = ["Rekomendasi Lagu", "Logout"]
            st.success(f"✅ Logged in as: {st.session_state.get('name', 'User')}")
        else:
            menu_options = ["Login", "Register"]
            if st.session_state.get("logout_clicked", False):
                st.info("🔓 Anda telah logout")
            else:
                st.info("🔒 Silakan login untuk mengakses fitur rekomendasi")
        
        menu = st.selectbox(
            "Pilih Menu:",
            menu_options,
            index=0
        )
    
    # Handle different menu options
    if menu == "Login":
        if not is_logged_in:
            handle_login()
        else:
            st.info("Anda sudah login. Pilih menu lain atau logout.")
    elif menu == "Register":
        if not is_logged_in:
            handle_register()
        else:
            st.info("Anda sudah login. Logout terlebih dahulu untuk register akun baru.")
    elif menu == "Rekomendasi Lagu":
        if is_logged_in:
            handle_recommendations()
        else:
            st.warning("Anda harus login terlebih dahulu untuk mengakses fitur ini.")
            st.info("Silakan pilih menu 'Login' di sidebar.")
    elif menu == "Logout":
        if is_logged_in or st.session_state.get("authentication_status") == True:
            handle_logout()
        else:
            st.info("Anda belum login.")

if __name__ == "__main__":
    main()