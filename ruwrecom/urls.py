from django.urls import path

from .views import *

urlpatterns = [
    path('ruwrecom/', postRecommendView),
    path('ruwrecom-rebuild/', setPostRecommendView),
    path('', homePageView, name='home'),
]