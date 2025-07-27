import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import pandas as pd
import os
from surprise import SVD, Dataset, Reader
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
    """Fixed content-based recommendation with proper similarity calculation"""
    print(f"CBF: Processing for user {user_id}")
    
    # Get liked songs for this user
    liked_songs = user_likes_df[user_likes_df['user_id'] == user_id]['song'].tolist()
    print(f"CBF: User liked {len(liked_songs)} songs: {liked_songs[:3]}...")
    
    if not liked_songs:
        print("CBF: No liked songs found")
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original dataframe
    songs_copy = songs_df.copy()
    
    # Get features for liked songs
    liked_features = songs_copy[songs_copy['song'].isin(liked_songs)]
    
    if liked_features.empty:
        print("CBF: No matching features found for liked songs")
        return pd.DataFrame()
    
    # Feature columns (make sure these exist in your dataset)
    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in songs_copy.columns]
    if missing_cols:
        print(f"CBF: Missing columns: {missing_cols}")
        # Use only available columns
        feature_cols = [col for col in feature_cols if col in songs_copy.columns]
    
    print(f"CBF: Using features: {feature_cols}")
    
    # Normalize features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(songs_copy[feature_cols])
    liked_features_scaled = scaler.transform(liked_features[feature_cols])
    
    # Create user profile (average of liked songs)
    user_profile = liked_features_scaled.mean(axis=0).reshape(1, -1)
    print(f"CBF: User profile shape: {user_profile.shape}")
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_profile, all_features_scaled)[0]
    print(f"CBF: Similarity range: {similarities.min():.4f} to {similarities.max():.4f}")
    
    # Add similarities to dataframe
    songs_copy['similarity'] = similarities
    
    # Filter out already liked songs and get top recommendations
    recommendations = songs_copy[~songs_copy['song'].isin(liked_songs)].copy()
    recommendations = recommendations.sort_values(by='similarity', ascending=False)
    
    # Get top N recommendations
    top_recs = recommendations[['song', 'artist', 'similarity']].head(top_n)
    top_recs = top_recs.rename(columns={"similarity": "estimated_rating"})
    
    print(f"CBF: Generated {len(top_recs)} recommendations")
    print(f"CBF: Score range: {top_recs['estimated_rating'].min():.4f} to {top_recs['estimated_rating'].max():.4f}")
    
    return top_recs

def collaborative_filtering_recommendation(user_id, df_songs, user_likes, top_n=10):
    """Fixed collaborative filtering with better error handling"""
    print(f"CF: Processing for user {user_id}")
    
    # Check if we have enough data for collaborative filtering
    unique_users = user_likes['user_id'].nunique()
    unique_songs = user_likes['song'].nunique()
    
    print(f"CF: Dataset has {unique_users} users and {unique_songs} songs")
    
    if unique_users < 2 or unique_songs < 2:
        print("CF: Not enough data for collaborative filtering")
        return pd.DataFrame()
    
    # Prepare rating data
    df_ratings = user_likes.copy()
    df_ratings["rating"] = 1.0  # Binary rating (liked = 1)
    
    # Add some noise to ratings to create variation
    # This helps when all ratings are the same
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.1, len(df_ratings))
    df_ratings["rating"] = df_ratings["rating"] + noise
    df_ratings["rating"] = df_ratings["rating"].clip(0.1, 1.0)  # Keep in valid range
    
    # Create Surprise dataset
    reader = Reader(rating_scale=(0.1, 1.0))
    data = Dataset.load_from_df(df_ratings[["user_id", "song", "rating"]], reader)
    
    # Split data
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train SVD model
    model = SVD(random_state=42, n_factors=10, n_epochs=20)
    model.fit(trainset)
    
    # Get songs user hasn't liked
    user_liked_songs = set(user_likes[user_likes["user_id"] == user_id]["song"])
    all_songs = set(df_songs["song"])
    unseen_songs = list(all_songs - user_liked_songs)
    
    print(f"CF: Predicting for {len(unseen_songs)} unseen songs")
    
    # Predict ratings for unseen songs
    predictions = []
    for song in unseen_songs:
        pred = model.predict(user_id, song)
        predictions.append((song, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_predictions = predictions[:top_n]
    
    # Create dataframe
    rec_df = pd.DataFrame(top_predictions, columns=["song", "estimated_rating"])
    
    # Add artist information
    final_recs = pd.merge(rec_df, df_songs[["song", "artist"]], on="song", how="left")
    
    print(f"CF: Generated {len(final_recs)} recommendations")
    if not final_recs.empty:
        print(f"CF: Score range: {final_recs['estimated_rating'].min():.4f} to {final_recs['estimated_rating'].max():.4f}")
    
    return final_recs

def hybrid_filtering(user_id, df_songs, user_likes, top_n=10):
    """Fixed hybrid filtering that ALWAYS returns consistent column structure"""
    print(f"HYBRID: Starting hybrid recommendation for user {user_id}")
    
    # Get recommendations from both methods
    cbf_recs = content_based_recommendation(user_id, df_songs, user_likes, top_n * 2)
    cf_recs = collaborative_filtering_recommendation(user_id, df_songs, user_likes, top_n * 2)
    
    print(f"HYBRID: CBF returned {len(cbf_recs)} recommendations")
    print(f"HYBRID: CF returned {len(cf_recs)} recommendations")
    
    # Handle fallback cases - ALWAYS ensure consistent column structure
    if cbf_recs.empty and cf_recs.empty:
        print("HYBRID: Both methods failed")
        return pd.DataFrame(columns=['song', 'artist', 'hybrid_score', 'method'])
    
    elif cbf_recs.empty:
        print("HYBRID: Using only CF recommendations")
        cf_fallback = cf_recs.head(top_n).copy()
        # Ensure we have the required columns
        if 'estimated_rating' in cf_fallback.columns:
            cf_fallback['hybrid_score'] = cf_fallback['estimated_rating']
        else:
            cf_fallback['hybrid_score'] = 0.5  # Default score
        cf_fallback['method'] = 'cf_only'
        # Return only the required columns in the correct order
        return cf_fallback[['song', 'artist', 'hybrid_score', 'method']].reset_index(drop=True)
    
    elif cf_recs.empty:
        print("HYBRID: Using only CBF recommendations")
        cbf_fallback = cbf_recs.head(top_n).copy()
        # Ensure we have the required columns
        if 'estimated_rating' in cbf_fallback.columns:
            cbf_fallback['hybrid_score'] = cbf_fallback['estimated_rating']
        else:
            cbf_fallback['hybrid_score'] = 0.5  # Default score
        cbf_fallback['method'] = 'cbf_only'
        # Return only the required columns in the correct order
        return cbf_fallback[['song', 'artist', 'hybrid_score', 'method']].reset_index(drop=True)
    
    # Both methods have results - proceed with hybrid combination
    print("HYBRID: Both methods have results, combining...")
    
    # Normalize scores to 0-1 range for fair combination
    def normalize_scores(df, score_col):
        if df.empty or df[score_col].std() == 0:
            return df
        df_copy = df.copy()
        min_score = df_copy[score_col].min()
        max_score = df_copy[score_col].max()
        if max_score != min_score:
            df_copy[score_col + '_normalized'] = (df_copy[score_col] - min_score) / (max_score - min_score)
        else:
            df_copy[score_col + '_normalized'] = 0.5  # Default middle value if all scores are the same
        return df_copy
    
    # Normalize scores
    cbf_recs = normalize_scores(cbf_recs, 'estimated_rating')
    cf_recs = normalize_scores(cf_recs, 'estimated_rating')
    
    # Create dictionaries for easy lookup
    cbf_scores = {}
    cf_scores = {}
    
    if 'estimated_rating_normalized' in cbf_recs.columns:
        cbf_scores = dict(zip(cbf_recs["song"], cbf_recs["estimated_rating_normalized"]))
    else:
        cbf_scores = dict(zip(cbf_recs["song"], cbf_recs["estimated_rating"]))
    
    if 'estimated_rating_normalized' in cf_recs.columns:
        cf_scores = dict(zip(cf_recs["song"], cf_recs["estimated_rating_normalized"]))
    else:
        cf_scores = dict(zip(cf_recs["song"], cf_recs["estimated_rating"]))
    
    # Get all unique recommended songs
    all_songs = set(cbf_scores.keys()).union(set(cf_scores.keys()))
    print(f"HYBRID: Combining {len(all_songs)} unique songs")
    
    # Calculate hybrid scores with weighted combination
    hybrid_scores = []
    cbf_weight = 0.6  # Give slightly more weight to content-based
    cf_weight = 0.4
    
    for song in all_songs:
        cbf_score = cbf_scores.get(song, 0)
        cf_score = cf_scores.get(song, 0)
        
        # Weighted average
        if song in cbf_scores and song in cf_scores:
            # Both methods recommend this song
            hybrid_score = (cbf_weight * cbf_score) + (cf_weight * cf_score)
            method = "both"
        elif song in cbf_scores:
            # Only content-based recommends
            hybrid_score = cbf_score * 0.8  # Slight penalty for single method
            method = "cbf_only"
        else:
            # Only collaborative filtering recommends
            hybrid_score = cf_score * 0.8  # Slight penalty for single method
            method = "cf_only"
        
        hybrid_scores.append((song, hybrid_score, method))
    
    # Sort by hybrid score
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_hybrid = hybrid_scores[:top_n]
    
    # Create result dataframe with EXACTLY the required structure
    result_data = []
    for song, score, method in top_hybrid:
        result_data.append({
            'song': song,
            'hybrid_score': score,
            'method': method
        })
    
    hybrid_df = pd.DataFrame(result_data)
    
    # Add artist information
    final_hybrid = pd.merge(hybrid_df, df_songs[["song", "artist"]], on="song", how="left")
    
    # ENSURE exact column order and structure
    final_hybrid = final_hybrid[['song', 'artist', 'hybrid_score', 'method']].reset_index(drop=True)
    
    print(f"HYBRID: Final recommendations with columns: {list(final_hybrid.columns)}")
    for i, row in final_hybrid.head().iterrows():
        print(f"  {row['song']} - {row['artist']}: {row['hybrid_score']:.4f} ({row['method']})")
    
    return final_hybrid

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
                # Parse song and artist
                titles = [s.rsplit(" - ", 1)[0] for s in selected_songs]
                artists = [s.rsplit(" - ", 1)[1] for s in selected_songs]
                
                # Create new likes dataframe
                new_likes = pd.DataFrame({
                    "user_id": [st.session_state["username"]] * len(selected_songs),
                    "song": titles,
                    "artist": artists,
                    "rating": [1.0] * len(selected_songs)
                })
                
                # Save to file
                updated_likes = pd.concat([user_likes, new_likes], ignore_index=True)
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