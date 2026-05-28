from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas
from pandas import DataFrame
import numpy as np
import random
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ruwrecom.services.recommender import (
    RecommenderService
)



# Create your views here.
def homePageView(request):
    # return HttpResponse("Hello, world!!!")
    return render(request, 'index.html')

@csrf_exempt
def postRecommendView(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=403)
    # posts = get_posts("F:\\MyFiles\\lenta-ru-news-recommend-data12.csv")
    # liked_indices = [78, 89, 95, 241, 268, 544, 567]
    # liked_posts = [posts[i] for i in liked_indices]
    data = json.loads(
        request.body
    )
    # print(data)
    # return JsonResponse({'status': True, 'message': 'Recommendation index bla'})
    liked_indices = data["likedIndices"]
    liked_posts = data["likedTexts"]

    recommender = RecommenderService()

    recommended_post_ids = recommender.recommend(

        liked_texts=liked_posts,

        liked_post_ids=liked_indices,

        top_k=100,

        feed_size=50,

        explore_probability=0.2
    )

    # res_posts = []
    # for score in recommended_post_ids[:50]:
    #     if score not in liked_indices:
    #         res_posts.append(posts[score])


    # data = {'scores': recommended_post_ids[:50], 'liked_indices': liked_indices, 'posts': res_posts}
    print(recommended_post_ids[:50])
    data = {'status': 'success', 'recommendations': recommended_post_ids[:50]}
    return JsonResponse(data)


@csrf_exempt
def setPostRecommendView(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=403)
    # posts = get_posts("F:\\MyFiles\\lenta-ru-news-recommend-data12.csv")
    # posts_with_ids = [{'id': k, 'text': v} for k, v in enumerate(posts)]

    data = json.loads(
        request.body
    )
    posts = data["posts"]
    recommender = RecommenderService()
    rebuild_result = recommender.rebuild(posts)
    return JsonResponse({'status': True, 'message': 'scheduled'})





TOKEN_RE = re.compile(r'[\w\d]+')
def tokenize_text_simple_regex(txt, min_token_size=4):

  txt = txt.lower()
  all_tokens = TOKEN_RE.findall(txt)

  return [token for token in all_tokens if len(token) >= min_token_size]



def get_posts(path: str):
    return ((pandas.read_csv(path))['text']).tolist()


