from django.http import HttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas
from pandas import DataFrame
import numpy as np
import random
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ruwrecom.services.recommender import (
    RecommenderService
)



# Create your views here.
def homePageView(request):
    # return HttpResponse("Hello, world!!!")
    return render(request, 'index.html')


def postRecommendView(request):

    posts = get_posts("F:\\MyFiles\\lenta-ru-news-recommend-data1.csv")
    liked_indices = [78, 89, 95, 241, 268, 544, 567]
    liked_posts = [posts[i] for i in liked_indices]

    recommender = RecommenderService()

    recommended_post_ids = recommender.recommend(

        liked_texts=liked_posts,

        liked_post_ids=liked_indices,

        top_k=100,

        feed_size=20,

        explore_probability=0.2
    )

    res_posts = []
    for score in recommended_post_ids[:50]:
        if score not in liked_indices:
            res_posts.append(posts[score])


    data = {'scores': recommended_post_ids[:50], 'liked_indices': liked_indices, 'posts': res_posts}
    return render(request, "ruwrecom.html", context=data)


def setPostRecommendView(request):
    posts = get_posts("F:\\MyFiles\\lenta-ru-news-recommend-data1.csv")
    posts_with_ids = [{'id': k, 'text': v} for k, v in enumerate(posts)]
    recommender = RecommenderService()
    recommender.rebuild(posts_with_ids)





TOKEN_RE = re.compile(r'[\w\d]+')
def tokenize_text_simple_regex(txt, min_token_size=4):

  txt = txt.lower()
  all_tokens = TOKEN_RE.findall(txt)

  return [token for token in all_tokens if len(token) >= min_token_size]



def get_posts(path: str):
    return ((pandas.read_csv(path))['text']).tolist()


