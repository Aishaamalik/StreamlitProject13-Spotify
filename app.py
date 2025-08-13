import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import io
from sklearn.inspection import permutation_importance
import functools

# Spotify color scheme
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"
SPOTIFY_LIGHT_GRAY = "#B3B3B3"
SPOTIFY_DARK_GRAY = "#282828"

st.set_page_config(
    page_title="Spotify Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎵"
)

# Centered main Spotify logo and app title
st.markdown(
    f"""
    <div style='display: flex; flex-direction: column; align-items: center; margin-bottom: 2rem;'>
        <img src="https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png" width="220"/>
        <h1 style='color:{SPOTIFY_GREEN}; margin-top: 0.5rem; font-size: 2.5rem; font-weight: bold; letter-spacing: 2px;'>
            Spotify Data Explorer
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS for Spotify theme
st.markdown(f"""
    <style>
    .main {{
        background-color: {SPOTIFY_BLACK};
        color: {SPOTIFY_WHITE};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {SPOTIFY_DARK_GRAY};
        border-radius: 10px 10px 0 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {SPOTIFY_WHITE};
        font-weight: 600;
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: {SPOTIFY_GREEN};
        color: {SPOTIFY_BLACK};
        border-radius: 10px 10px 0 0;
    }}
    .stButton>button {{
        background-color: {SPOTIFY_GREEN};
        color: {SPOTIFY_BLACK};
        border-radius: 8px;
        font-weight: bold;
    }}
    .stAlert {{
        background-color: {SPOTIFY_DARK_GRAY};
        color: {SPOTIFY_GREEN};
        border-left: 5px solid {SPOTIFY_GREEN};
        border-radius: 8px;
        padding: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("spotify.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

df = load_data()

if df is None or df.empty:
    st.error("No data available. Please check the CSV file.")
    st.stop()

# Clean column names
cols = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
df.columns = cols

# After loading and cleaning df
if 'Artist_Name' in df.columns:
    df['Artist_Name'] = df['Artist_Name'].astype('category')
if 'Song_Name' in df.columns:
    df['Song_Name'] = df['Song_Name'].astype('category')

def get_top_n(series, n=100):
    return list(series.value_counts().head(n).index)

# Sidebar Filters (optimized)
st.sidebar.title("Spotify Data Explorer")
st.sidebar.markdown(
    f"<span style='color:{SPOTIFY_GREEN};font-weight:bold;'>Modern EDA, ML & Export</span>", unsafe_allow_html=True
)
st.sidebar.header("Filters", divider='rainbow')
select_all = st.sidebar.checkbox("Select All Data", value=True)

# Top 100 artists/songs for performance
with st.spinner("Preparing filter options..."):
    top_artists = get_top_n(df['Artist_Name'], 100)
    top_songs = get_top_n(df['Song_Name'], 100)

st.sidebar.markdown("<i>Only top 100 most frequent artists/songs shown for performance.</i>", unsafe_allow_html=True)
selected_artists = st.sidebar.multiselect(
    "Artist(s)", top_artists, default=top_artists if select_all else [], help="Type to search. Only top 100 shown."
)
artist_search = st.sidebar.text_input("Or search artist (exact match):", "")
if artist_search and artist_search not in selected_artists:
    selected_artists.append(artist_search)

selected_songs = st.sidebar.multiselect(
    "Song(s)", top_songs, default=top_songs if select_all else [], help="Type to search. Only top 100 shown."
)
song_search = st.sidebar.text_input("Or search song (exact match):", "")
if song_search and song_search not in selected_songs:
    selected_songs.append(song_search)

# Numeric filters (optimized)
st.sidebar.markdown("<b>Numeric Filters</b>", unsafe_allow_html=True)
num_cols = df.select_dtypes(include=[np.number]).columns
selected_num_cols = st.sidebar.multiselect(
    "Select numeric columns to filter:", list(num_cols), default=[])
num_filters = {}
for col in selected_num_cols:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    if max_val - min_val > 1_000_000:
        st.sidebar.warning(f"Range for {col} is very large. Slider may be hard to use.")
    val = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
    num_filters[col] = val

# Apply filters
df_filtered = df.copy()
if not select_all:
    if selected_artists:
        df_filtered = df_filtered[df_filtered['Artist_Name'].isin(selected_artists)]
    if selected_songs:
        df_filtered = df_filtered[df_filtered['Song_Name'].isin(selected_songs)]
for col, (minv, maxv) in num_filters.items():
    df_filtered = df_filtered[(df_filtered[col] >= minv) & (df_filtered[col] <= maxv)]

# Show warning for deprecated use_column_width
USE_COLUMN_WIDTH_DEPRECATED = False

def safe_dataframe(*args, **kwargs):
    if 'use_column_width' in kwargs:
        global USE_COLUMN_WIDTH_DEPRECATED
        USE_COLUMN_WIDTH_DEPRECATED = True
        kwargs['use_container_width'] = kwargs.pop('use_column_width')
    return st.dataframe(*args, **kwargs)

def safe_sidebar_image(*args, **kwargs):
    if 'use_column_width' in kwargs:
        global USE_COLUMN_WIDTH_DEPRECATED
        USE_COLUMN_WIDTH_DEPRECATED = True
        kwargs['use_container_width'] = kwargs.pop('use_column_width')
    return st.sidebar.image(*args, **kwargs)

# Sidebar
safe_sidebar_image(
    "https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png",
    use_container_width=True
)
# Sidebar Filters
st.sidebar.title("Spotify Data Explorer")
st.sidebar.markdown(
    f"<span style='color:{SPOTIFY_GREEN};font-weight:bold;'>Modern EDA, ML & Export</span>", unsafe_allow_html=True
)

# Tabs
tabs = st.tabs([
    "Data Overview",
    "Data Visualizations",
    "EDA",
    "Statistical Analysis",
    "Model Application",
    "Model Interpretation",
    "Export"
])

# --- Data Overview ---
with tabs[0]:
    st.header("Data Overview")
    st.write(f"**Shape:** {df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns")
    safe_dataframe(df_filtered.head(20), use_container_width=True, height=400)
    st.write("**Column Info:**")
    safe_dataframe(pd.DataFrame({
        'Type': df_filtered.dtypes,
        'Missing': df_filtered.isnull().sum(),
        'Unique': df_filtered.nunique()
    }))
    st.write("**Summary Statistics:**")
    safe_dataframe(df_filtered.describe(include='all').T)
    st.info("Tip: Use the sidebar to filter data. All visualizations and models use the filtered data.")

# --- Data Visualizations ---
with tabs[1]:
    st.header("Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Artists by Total Streams")
        top_artists = df_filtered.groupby('Artist_Name')['Total_Streams'].sum().sort_values(ascending=False).head(10)
        # Escape $ in artist names to prevent mathtext parsing
        escaped_labels = [str(label).replace('$', '\\$') for label in top_artists.index]
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(y=escaped_labels, x=top_artists.values, ax=ax, palette=[SPOTIFY_GREEN]*10)
        ax.set_xlabel("Total Streams")
        ax.set_ylabel("")
        st.pyplot(fig)
    with col2:
        st.subheader("Distribution of Peak Streams")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df_filtered['Peak_Streams'], bins=30, color=SPOTIFY_GREEN)
        ax.set_xlabel("Peak Streams")
        st.pyplot(fig)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_filtered[num_cols].corr(), annot=True, cmap="Greens")
    st.pyplot(fig)
    st.info("Tip: Visualizations update with filters.")

# --- EDA ---
with tabs[2]:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("**Outlier Detection (Peak Streams):**")
    q1 = df_filtered['Peak_Streams'].quantile(0.25)
    q3 = df_filtered['Peak_Streams'].quantile(0.75)
    iqr = q3 - q1
    outliers = df_filtered[(df_filtered['Peak_Streams'] < q1 - 1.5*iqr) | (df_filtered['Peak_Streams'] > q3 + 1.5*iqr)]
    st.write(f"Outliers: {outliers.shape[0]} rows")
    safe_dataframe(outliers.head(10), use_container_width=True)
    st.write("**Feature Relationships:**")
    st.write("Scatterplot: Total Streams vs. Peak Streams")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='Peak_Streams', y='Total_Streams', data=df_filtered, color=SPOTIFY_GREEN)
    st.pyplot(fig)

# --- Statistical Analysis ---
with tabs[3]:
    st.header("Statistical Analysis")
    st.write("**Correlation Matrix:**")
    safe_dataframe(df_filtered[num_cols].corr(), use_container_width=True)
    st.write("**Group Statistics (by Artist):**")
    safe_dataframe(df_filtered.groupby('Artist_Name')['Total_Streams'].agg(['mean','sum','count']).sort_values('sum', ascending=False).head(10), use_container_width=True)

# --- Model Application ---
with tabs[4]:
    st.header("Model Application")
    st.write("Select a target column for regression or classification.")
    model_type = st.radio("Model Type", ["Regression", "Classification"], horizontal=True)
    if model_type == "Regression":
        target = st.selectbox("Target (Regression)", [c for c in num_cols if c not in ['Position']])
        features = st.multiselect("Features", [c for c in num_cols if c != target and c != 'Position'])
        if not features or not target:
            st.warning("Please select at least one feature and a target for regression.")
        else:
            X = df_filtered[features].fillna(0)
            y = df_filtered[target].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'Top Feature Only': LinearRegression()
            }
            results = {}
            for name, model in models.items():
                if name == 'Top Feature Only':
                    top_feat = X_train.corrwith(y_train).abs().idxmax()
                    model.fit(X_train[[top_feat]], y_train)
                    y_pred = model.predict(X_test[[top_feat]])
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[name] = {'MSE': mse, 'R2': r2}
            st.write(pd.DataFrame(results).T)
            best_model_name = max(results, key=lambda k: results[k]['R2'])
            st.success(f"Best Regression Model: {best_model_name}")
            st.session_state['best_model'] = models[best_model_name]
            st.session_state['model_type'] = 'regression'
            st.session_state['features'] = features
            st.session_state['target'] = target
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
    else:
        if 'Top_10_xTimes' in df_filtered.columns:
            df_filtered['is_top10'] = (df_filtered['Top_10_xTimes'] > 0).astype(int)
            target = 'is_top10'
            features = st.multiselect("Features", [c for c in num_cols if c != 'Position'])
            if not features:
                st.warning("Please select at least one feature for classification.")
            else:
                X = df_filtered[features].fillna(0)
                y = df_filtered[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Top Feature Only': LogisticRegression(max_iter=1000)
                }
                results = {}
                for name, model in models.items():
                    if name == 'Top Feature Only':
                        top_feat = X_train.corrwith(y_train).abs().idxmax()
                        model.fit(X_train[[top_feat]], y_train)
                        y_pred = model.predict(X_test[[top_feat]])
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results[name] = {'Accuracy': acc}
                st.write(pd.DataFrame(results).T)
                best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
                st.success(f"Best Classification Model: {best_model_name}")
                st.session_state['best_model'] = models[best_model_name]
                st.session_state['model_type'] = 'classification'
                st.session_state['features'] = features
                st.session_state['target'] = target
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
        else:
            st.warning("No suitable column for classification target found.")

# --- Model Interpretation ---
with tabs[5]:
    st.header("Model Interpretation")
    if 'best_model' in st.session_state:
        model = st.session_state['best_model']
        features = st.session_state['features']
        target = st.session_state['target']
        X_test = st.session_state.get('X_test')
        y_test = st.session_state.get('y_test')
        st.subheader("Model Performance Summary")
        if st.session_state['model_type'] == 'regression':
            y_pred = model.predict(X_test)
            st.metric("R2 Score", round(r2_score(y_test, y_pred), 3))
            st.metric("MSE", round(mean_squared_error(y_test, y_pred), 2))
        else:
            y_pred = model.predict(X_test)
            st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
            st.text(classification_report(y_test, y_pred))
        st.subheader("Sample Predictions")
        sample_df = X_test.copy()
        sample_df['True'] = y_test
        sample_df['Predicted'] = y_pred
        safe_dataframe(sample_df.head(10), use_container_width=True)
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importances:**")
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            safe_dataframe(imp_df.sort_values('Importance', ascending=False), use_container_width=True)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x='Importance', y='Feature', data=imp_df, palette=[SPOTIFY_GREEN]*len(features))
            st.pyplot(fig)
        elif hasattr(model, 'coef_'):
            st.write("**Model Coefficients:**")
            coefs = model.coef_.flatten() if hasattr(model.coef_, 'flatten') else model.coef_
            coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
            safe_dataframe(coef_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=[SPOTIFY_GREEN]*len(features))
            st.pyplot(fig)
        else:
            st.write("**Permutation Importances:**")
            try:
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                perm_df = pd.DataFrame({'Feature': features, 'Importance': result.importances_mean})
                safe_dataframe(perm_df.sort_values('Importance', ascending=False), use_container_width=True)
                fig, ax = plt.subplots(figsize=(8,4))
                sns.barplot(x='Importance', y='Feature', data=perm_df, palette=[SPOTIFY_GREEN]*len(features))
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Model does not support feature importances or permutation importances. ({e})")
    else:
        st.warning("No model trained yet. Please train a model in the previous tab.")

# --- Export ---
with tabs[6]:
    st.header("Export Section")
    st.write("Download the best model and cleaned data.")
    if 'best_model' in st.session_state:
        buffer = io.BytesIO()
        pickle.dump(st.session_state['best_model'], buffer)
        buffer.seek(0)
        st.download_button(
            label="Download Best Model (pickle)",
            data=buffer,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )
    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="spotify_cleaned.csv",
        mime="text/csv"
    )
    st.write("**Code Snippet to Load Model:**")
    st.code("""
import pickle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    # Use model.predict(X) for inference
""", language="python")
    st.success("App is ready to use! All guidelines followed. Enjoy exploring your Spotify data.")