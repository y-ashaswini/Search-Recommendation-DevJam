from fastapi import FastAPI
import pandas as pd
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fastapi.middleware.cors import CORSMiddleware
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt

app = FastAPI()
data_path = 'devjam_data1.csv'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "CONNECTED"}


@app.get("/search_results/{search_param}")
async def get_search_results(search_param: str):
    df = pd.read_csv(data_path)

    df = df.fillna(df['Period'].mean())

    df = df.astype({'Period': int})
    df = df.astype({'Period': str})

    for i in range(0, len(df["Type"])):
        if "Painting" in df.iat[i, 4]:
            df.iat[i, 4] = "Painting"

    columns = df.columns

    for index, row in df.iterrows():

        for col in columns:
            if col != 'ID':
                row[col] = row[col].lower()
            else:
                pass

    df['Key_words'] = ""

    for index, row in df.iterrows():
        plot = row['Description']
        r = Rake()
        r.extract_keywords_from_text(plot)
        key_words_dict_scores = r.get_word_degrees()
        row['Key_words'] = list(key_words_dict_scores.keys())

    df.drop(columns=['Description'], inplace=True)

    df.set_index('ID', inplace=True)

    df['bag_of_words'] = ''
    columns = df.columns
    for index, row in df.iterrows():
        words = ''
        for col in columns:
            if col != 'Origin' and col != 'ID' and col != 'Period' and col != 'Type' and col != 'Name':
                words = words + ' '.join(row[col]) + ' '
            elif col != 'ID':
                words = words + row[col] + ' '
        row['bag_of_words'] = words

    df.drop(columns=[col for col in df.columns if col != 'bag_of_words' and col != 'ID'], inplace=True)

    count = CountVectorizer()
    count_matrix = count.fit_transform(df['bag_of_words'])

    indices = pd.Series(df.index)

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    def recommendations(keyword, cosine_sim=cosine_sim):

        recommended_arts = []

        keyword = keyword.lower()
        keyword = keyword.strip()

        for i in range(0, len(df['bag_of_words'])):
            terms = df.iat[i, 0]
            if keyword in terms:
                art_id = df.index[df['bag_of_words'] == df.iat[i, 0]].tolist()[0]
                break
            else:
                pass

        idx = indices[indices == art_id].index[0]

        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

        top_10_indexes = list(score_series.iloc[:11].index)

        for i in top_10_indexes:
            recommended_arts.append(list(df.index)[i])

        return recommended_arts

    return {"search_results": recommendations(search_param)}
