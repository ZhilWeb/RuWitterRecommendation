from django.urls import path

from .views import *

urlpatterns = [
    path('ruwrecom/', postRecommendView),
    path('', homePageView, name='home'),
]