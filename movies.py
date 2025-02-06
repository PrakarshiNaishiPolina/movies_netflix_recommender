import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Page Config
st.set_page_config(page_title="Netflix Recommendations", page_icon="📽️")

# Load the dataset
df=pd.read_csv("netflix_titles.csv")

# Filling the missing values with ""
df.fillna("",inplace=True)

print(df.head())

# Splits the data into movies and tv shows- two DataFrames and creates a new independent copy so that modifications to df_movies or df_tvshows do not alter the original df.

df_movies=df[df["type"]=="Movie"].copy()
df_tvshows=df[df["type"]=="TV Show"].copy()

# Encoding categorical features
encoder_genre_movies = LabelEncoder()
encoder_genre_tvshows = LabelEncoder()
encoder_cast_movies = LabelEncoder()
encoder_cast_tvshows = LabelEncoder()

# Removes rows where genre or cast is empty

df_movies = df_movies[df_movies["listed_in"] != ""]
df_tvshows = df_tvshows[df_tvshows["listed_in"] != ""]
df_movies = df_movies[df_movies["cast"] != ""]
df_tvshows = df_tvshows[df_tvshows["cast"] != ""]

df_movies["genre_encoded"] = encoder_genre_movies.fit_transform(df_movies["listed_in"])
df_tvshows["genre_encoded"] = encoder_genre_tvshows.fit_transform(df_tvshows["listed_in"])

df_movies["cast_encoded"] = encoder_cast_movies.fit_transform(df_movies["cast"])
df_tvshows["cast_encoded"] = encoder_cast_tvshows.fit_transform(df_tvshows["cast"])


# Streamlit

st.title("🎬 Netflix Recommendation System")

# User inputs
user_type = st.radio("Select Type:", ["Movie", "TV Show"])

if user_type == "Movie":
    df_filtered = df_movies
    encoder_genre = encoder_genre_movies
    encoder_cast = encoder_cast_movies
else:
    df_filtered = df_tvshows
    encoder_genre = encoder_genre_tvshows
    encoder_cast = encoder_cast_tvshows

genres_list = df_filtered["listed_in"].unique().tolist() # after getting unique values in the form of array we convert it into list.
cast_list = df_filtered["cast"].unique().tolist()

user_genre = st.selectbox("Select Genre:", genres_list, index=0 if genres_list else None)
user_cast = st.selectbox("Select Cast:", cast_list, index=0 if cast_list else None)

# index =9 selects the first available option in the dropdown

def recommend_shows(user_genre, user_cast):
    # Check if user inputs exist in the trained LabelEncoders
    if user_genre not in encoder_genre.classes_ or user_cast not in encoder_cast.classes_:
        return pd.DataFrame()  # Return empty DataFrame if the input isn't valid

    user_genre_encoded = encoder_genre.transform([user_genre])[0]
    user_cast_encoded = encoder_cast.transform([user_cast])[0]

    # Filter based on encoded values
    filtered_df = df_filtered[
        (df_filtered["genre_encoded"] == user_genre_encoded) &
        (df_filtered["cast_encoded"] == user_cast_encoded)
    ]

    # Select top 3 recommendations
    return filtered_df[["title", "listed_in", "cast", "description"]].head(3)


# Button to generate recommendations

if st.button("Get Recommendations"):
    recommendations = recommend_shows(user_genre, user_cast)
    if not recommendations.empty:
        st.subheader("📌 Top 3 Recommendations:")
        st.dataframe(recommendations)
    else:
        st.warning("No exact match found. Try different selections.")

# CSS for Streamlit
st.markdown(
    """
    <style>
        /* Background and text color */
        body, .stApp {
            background-color: #e8eaf6;  /* Soft grey-blue */
            color: #333333;  /* Dark grey for readability */
        }

        /* Style the title */
        .stTitle {
            color: #222;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }

        /* Style the select boxes */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff;
            border: 2px solid #aaa; 
            border-radius: 8px;
            padding: 6px;
        }

        /* Style the radio buttons */
        div[data-testid="stRadio"] label {
            background-color: #f0f0f5; /* Light grey for unselected */
            border-radius: 10px;
            padding: 8px 15px;
            margin: 4px;
            display: block;
            text-align: center;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }

        div[data-testid="stRadio"] label:hover {
            background-color: #d0d4f5; /* Light blue on hover */
        }

        div[data-testid="stRadio"] input:checked + label {
            background-color: #0056b3 !important; /* Dark blue for selected */
            color: white !important;
            border: 2px solid #003d82;
        }

        /* Style the button */
        .stButton>button {
            background-color: #0056b3;  /* Soft dark blue */
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 15px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #003d82; /* Slightly darker blue on hover */
        }

        /* Style the warning message */
        .stWarning {
            background-color: #fff3cd;
            border-left: 5px solid #ffae42;
            padding: 10px;
            color: #856404;
            border-radius: 5px;
        }

        /* Style DataFrame display */
        .stDataFrame {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }

    </style>
    """,
    unsafe_allow_html=True
)

