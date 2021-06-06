import re
import os
import string
import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


def fix_checks(check):
    try:
        return int(check)
    except:
        return np.nan




def check_text(check):
    if check<=800:
        return 'very cheap'
    elif check <=1500:
        return "moderate"
    elif check<=2000:
        return "higher than average"
    elif check <=3000:
        return "expensive"
    else:
        return "very expensive"




def prep_restaurant_data(restaurants_df):

    restaurants_df["check"] = restaurants_df["check"].apply(
        lambda row: fix_checks(check=row)
    )

    restaurants_df = restaurants_df.dropna()

    restaurants_df["check"] = restaurants_df["check"].apply(
        lambda row: check_text(check=row)
    )

    restaurants_df["restaurant_id"] = restaurants_df["url"].apply(
        lambda row: row.split("https://www.restoclub.ru/spb/place/")[1]
    )

    restaurants_df = (
        restaurants_df.copy()
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(columns={"index": "id"})[
            [
                "restaurant_id",
                "id",
                "name",
                "url",
                "short_desc",
                "num_reviews",
                "rating_overall",
                "rating_word",
                "cuisine",
                "address",
                "time",
                "check",
                "tags",
                "desc",
                "kitchen_rating",
                "interior_rating",
                "service_rating",
            ]
        ]
    )

    # mark users inputed restaurant
    restaurants_df["id"] = restaurants_df.apply(
        lambda row: row["id"] if row["restaurant_id"] != "users_input" else 1713, axis=1
    )

    # add text field combining tags and cuisine:

    restaurants_df["description"] = restaurants_df.apply(
        lambda row: " ".join([row["cuisine"], row["tags"]]),
        axis=1,
    )

    return restaurants_df




def create_similarity_matrix(restaurants_df):
    ds = restaurants_df.copy()
    ds = ds[['id','description']]

    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=0, stop_words="english"
    )
    tfidf_matrix = tf.fit_transform(ds["description"])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [
            (cosine_similarities[idx][i], ds["id"][i]) for i in similar_indices
        ]

        results[row["id"]] = similar_items[1:]

    return results#, cosine_similarities




# Just reads the results out of the dictionary.
def recommend(item_id, num, restaurants_df):
    
    results = create_similarity_matrix(restaurants_df)
    recs = results[item_id][:num]
    
    return recs




def get_recs_df(recs, restaurants_df):

    df_out = pd.DataFrame()

    for rec in recs:
        rec_index = rec[1]
        rec_restaurant_data = restaurants_df[restaurants_df["id"] == rec_index]
        df_out = pd.concat([df_out, rec_restaurant_data])

    df_out = df_out[
        [
            "restaurant_id",
            "name",
            "id",
            "url",
            "short_desc",
            "num_reviews",
            "rating_overall",
            "rating_word",
            "cuisine",
            "address",
            "time",
            "check",
            "tags",
            "desc",
            "kitchen_rating",
            "interior_rating",
            "service_rating",
        ]
    ]

    return df_out




def get_recommendations(item_id, restaurants_df, n_items=None):

    restaurant_df_checked = restaurants_df[
        restaurants_df["check"] == restaurants_df.iloc[item_id]["check"]
    ]
    num = len(restaurant_df_checked)
    
    if n_items == None:
        n_items = len(restaurant_df_checked)
    else:
        n_items = n_items

    recs = recommend(item_id, num, restaurants_df)
    recs = [i for i in recs if i[1]!=item_id][:n_items]

    return recs




def CB_recommender(
    user_id,
    ratings_df,
    n_items,
    restaurants_df,
    users_cuisine=None,
    users_tags=None,
    users_check=None,
):
    """ Takes the user_id, determines his favorite rest and recommends n_items similar to it """

    # understand users_pref
    if user_id in ratings_df["userID"].unique().tolist():
        favorite = ratings_df[ratings_df["userID"] == user_id].sort_values(
            "rating", ascending=False
        )

        favorite_rest = favorite.iloc[0]["itemID"].astype(int)
        recs = get_recommendations(
            item_id=favorite_rest, restaurants_df=restaurants_df, n_items=n_items
        )
    else:
        users_rest = pd.DataFrame(
            data={
                "name": [""],
                "url": ["https://www.restoclub.ru/spb/place/users_input"],
                "short_desc": [""],
                "num_reviews": [""],
                "rating_overall": [""],
                "rating_word": [""],
                "cuisine": [users_cuisine],
                "address": [""],
                "time": [""],
                "check": [users_check],
                "tags": [users_tags],
                "desc": [""],
                "kitchen_rating": [""],
                "interior_rating": [""],
                "service_rating": [""],
            }
        )

        users_rest = prep_restaurant_data(restaurants_df=users_rest)

        recs = get_recommendations(
            item_id=users_rest.iloc[0]['id'].astype(int),
            restaurants_df=pd.concat([restaurants_df, users_rest]).reset_index(drop=True),
            n_items=n_items,
        )

    result = get_recs_df(recs, restaurants_df)

    return result



def fix_reviews(reviews_df, restaurants_df):

    reviews_df["user_id"] = reviews_df["user_url"].apply(
        lambda row: row.split("https://www.restoclub.ru/user/")[1]
    )
    reviews_df["restaurant_id"] = reviews_df["restaurant_url"].apply(
        lambda row: row.split("https://www.restoclub.ru/spb/place/")[1]
    )

    # create ratings df:

    ratings = (
        reviews_df[["user_id", "restaurant_id", "rating_overall"]]
        .merge(restaurants_df[["restaurant_id", "id"]])[
            ["user_id", "id", "rating_overall"]
        ]
        .rename(columns={"rating_overall": "rating"})
        .drop_duplicates(["user_id", "id"])
        .rename(columns={"id": "itemID", "user_id": "userID"})
    )

    ratings.userID = ratings.userID.astype(int)

    ratings = ratings.dropna()

    return ratings



def get_visited(user_id, ratings_df):
    visited = ratings_df[ratings_df["userID"] == user_id]["itemID"].tolist()
    return visited


def create_algo():

    algo = SVD()

    # for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)

    return algo




def CF_recommender(user_id, ratings_df, n_items, restaurants_df):

    algo = create_algo()
    item_ids = restaurants.index
    ratings_pred = [algo.predict(user_id, i).est for i in item_ids]

    df_user = pd.DataFrame(data={"item_id": item_ids, "rating_pred": ratings_pred})

    # sort descending
    df_user = df_user.sort_values("rating_pred", ascending=False)

    # filter visited:
    df_user = df_user[~df_user["item_id"].isin(get_visited(user_id, ratings_df))]

    df_user = df_user[:n_items]
    
    result = restaurants_df[restaurants_df["id"].isin(df_user['item_id'].tolist())]

    return result #df_user




def recommendation_system(
    user_id, 
    restaurants_df, 
    reviews_df, 
    n_items, 
    users_cuisine=None,
    users_tags=None,
    users_check=None):

    ratings_df = fix_reviews(reviews_df=reviews_df, restaurants_df=restaurants_df)
    results = create_similarity_matrix(restaurants_df=restaurants_df)
    if user_id in ratings_df["userID"].unique().tolist():
        CF_recs = CF_recommender(user_id, ratings_df, n_items, restaurants_df)
        CB_recs = CB_recommender(user_id, ratings_df, n_items, restaurants_df, 
                                 users_cuisine=None, users_tags=None, users_check=None)
        all_recs = (
            pd.concat([CB_recs, CF_recs]).sample(frac=1)[:n_items].reset_index(drop=True)
        )
        
    else:
        all_recs = CB_recommender(
            user_id=user_id,
            ratings_df=ratings_df,
            n_items=n_items,
            restaurants_df=restaurants_df,
            users_cuisine=users_cuisine,
            users_tags=users_tags,
            users_check=users_check,
)
    all_recs = all_recs[[
        "name", "address", "cuisine", "check", "short_desc", "tags", "time"
    ]].reset_index(drop=True)


    return all_recs



# read restaurants data:
restaurants = pd.read_csv("restaurants_info.csv",index_col=0)
restaurants = prep_restaurant_data(restaurants_df=restaurants)


# read reviews_data:
reviews = pd.read_csv("reviews_info.csv",index_col = 0 )
ratings = fix_reviews(reviews_df=reviews, restaurants_df=restaurants)


reader = Reader(rating_scale=(1, 10))

# Loads Pandas dataframe
data = Dataset.load_from_df(ratings[['userID', 'itemID', 'rating']], reader)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)


import streamlit as st

st.set_page_config(layout='wide', initial_sidebar_state='auto')


st.write("# Get Your Restaurant Recommendations!")


st.sidebar.header("If you are new to the system, please share your preferences")


tags_list = ['кулинарные мастер-классы',
 'можно с животными',
 'детские мастер-классы',
 'доставка',
 'детская комната',
 'боулинг',
 'бранч',
 'заказ кейтеринга',
 'спортивные трансляции',
 'панорамный вид',
 'гастрономические сеты',
 'детское меню',
 'dj',
 'бильярд',
 'кабинки',
 'караоке',
 'кальян',
 'стриптиз',
 'бизнес-ланч',
 'за городом',
 'завтрак',
 'живая музыка',
 'настольные игры',
 'еда навынос',
 'ресторан у воды',
 'своя пивоварня',
 'можно со своей едой',
 'парковка',
 'здесь танцуют',
 'здесь живут котики',
 'камин',
 'при отеле',
 'шоу-программа']




cuisine_list = [
 'смешанная',
 'азиатская',
 'чешская',
 'украинская',
 'японская',
 'испанская',
 'мексиканская',
 'греческая',
 'сербская',
 'латиноамериканская',
 'авторская',
 'кавказская',
 'китайская',
 'стритфуд',
 'шашлыки',
 'стейки',
 'бургеры',
 'боулы',
 'грузинская',
 'шаверма',
 'средиземноморская',
 'тайская',
 'рыба и морепродукты',
 'блюда из дичи',
 'крабы',
 'индийская',
 'европейская',
 'узбекская',
 'пицца',
 'суши',
 'крафтовое пиво',
 'скандинавская',
 'итальянская',
 'восточная',
 'рамен',
 'армянская',
 'израильская',
 'вьетнамская',
 'десерты',
 'немецкая',
 'американская',
 'французская',
 'корейская',
 'бельгийская',
 'вегетарианская',
 'русская',
 'устрицы',
 'коктейли',
 'торты на заказ']

user_id = st.text_input("Enter your user ID: \n", "")

if user_id!="":
    user_id = int(float(user_id)) 
    if user_id not in ratings["userID"].unique().tolist():
        st.write("""
        #### You seem to be a new user. Please identify your preferences in the menu on the left.
        """)
    else:
        st.write(f"""
        #### Welcome back, user_{user_id}!
        """)

st.write("## ")

n_items = st.slider("How many recommendations do you want to see?", 1, 50, 1)


users_check = st.sidebar.slider(
    label="Choose your budget:",
    min_value=0,
    max_value=8000,
    step=500, value=None)  

users_cuisine = st.sidebar.multiselect('Select up to 3 words for the cuisine you want:',cuisine_list)
users_cuisine = ", ".join(users_cuisine)

users_tags = st.sidebar.multiselect('Select up to 3 words that best describe your place:',tags_list)
users_tags = ", ".join(users_tags)



if st.button("Get recommendations"):
    
    'Generating your recommendations...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        latest_iteration.text(f'Loaded {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.05)


    output_table = recommendation_system(
        user_id = user_id, 
        restaurants_df=restaurants, 
        reviews_df=reviews, 
        n_items=n_items, 
        users_cuisine=users_cuisine,
        users_tags=users_tags,
        users_check=users_check
    )
    
    st.table(output_table.assign(hack='').set_index('hack'))



