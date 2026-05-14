from django.http import HttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas
from pandas import DataFrame
import numpy as np


# Create your views here.
def homePageView(request):
    # return HttpResponse("Hello, world!!!")
    return render(request, 'index.html')


def postRecommendView(request):
    # if request.method != "POST":
    #     return HttpResponse(f"{request.method} method not supported")

    # liked_posts = []
    posts = get_posts("F:\\MyFiles\\lenta-ru-news-recommend-data1.csv")


    liked_indices = 545
    # liked_posts = []
    # for index in liked_indices:
    #     liked_posts.append(posts[index])
    # data = {'posts': liked_posts}
    # return render(request, "ruwrecom.html", context=data)

    # for i in liked_posts:
    #     posts.append(i)
    #     liked_indices.append(len(posts) - 1)

    vectorizer = TfidfVectorizer(tokenizer=tokenize_text_simple_regex, max_df=0.8, min_df=5)
    X = vectorizer.fit_transform(posts)


    # Усредняем векторы
    user_profile = X[liked_indices]

    # Считаем похожесть со всеми постами
    similarities = cosine_similarity(user_profile, X).flatten()

    # Рекомендации
    recommended = similarities.argsort()[::-1]

    data = {'posts': recommended[:50], 'liked_indices': liked_indices}
    return render(request, "ruwrecom.html", context=data)






TOKEN_RE = re.compile(r'[\w\d]+')
def tokenize_text_simple_regex(txt, min_token_size=4):

  txt = txt.lower()
  all_tokens = TOKEN_RE.findall(txt)

  return [token for token in all_tokens if len(token) >= min_token_size]



def get_posts(path: str):
    return ((pandas.read_csv(path))['text']).tolist()