from django.apps import AppConfig


class RuwrecomConfig(AppConfig):
    name = 'ruwrecom'

    def ready(self):
        from ruwrecom.services.recommender import (
            RecommenderService
        )

        recommender = RecommenderService()

        if recommender.state is None:
            recommender.initialize([])
