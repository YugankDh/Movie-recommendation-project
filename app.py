# Data Processing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

df = pd.read_csv("imdb-movies-dataset.csv")
df1 = df.copy()

#Processing the ml dataframe
df.drop(['Poster','Review Count','Review Title','Certificate','Metascore'],axis='columns',inplace=True)


df.isna().sum()

df[['Review','Cast','Director','Genre']] = df[['Review','Cast','Director','Genre']].fillna("")

df[['Year','Duration (min)','Votes']] = df[['Year','Duration (min)','Votes']].fillna(0)



df.isna().sum()

df['combined_text'] =   df['Genre'] + " " + df['Director'] + " " + df['Cast'] + " " + df['Review']

df['Votes'].apply(type).value_counts()

tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=2)
combined_vector = tfidf.fit_transform(df['combined_text'])

#processing raw data for output
df1['Votes'] = (
    df['Votes']
    .astype(str)
    .str.replace(',', '', regex=False)
    .astype(int)
)

df1[['Review','Cast','Director','Genre']] = df1[['Review','Cast','Director','Genre']].fillna("Not available")

df1[['Year','Duration (min)','Votes']] = df1[['Year','Duration (min)','Votes']].fillna("Not available")


df1['Rating'] = df1['Rating'].fillna(round(df1['Rating'].mean(),1))



st.header("🎬 Movie Recommendation System")
st.space("small")
user_input = st.text_input("Enter any keyword ",placeholder="Year / Genre / Cast / Director")

if st.button("Recommend movies"):
    if isinstance(user_input, str) and user_input.strip():
        input_tfidf = tfidf.transform([user_input])
        similarities = cosine_similarity(input_tfidf, combined_vector)
        top_indices = similarities[0].argsort()[::-1][:10]
        results = df1.iloc[top_indices][['Title','Year','Review','Director','Rating','Cast']]
        st.data_editor(results, hide_index=True)
        
    else:
        st.info("Type a keyword like actor, director, genre, or movie theme")



# imporve ui
# improve performance by wrapping processing in a function and using cache data on streamlit
