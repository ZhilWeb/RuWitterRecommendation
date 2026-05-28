import random
import threading

from dataclasses import dataclass

import hnswlib
import numpy as np

from sentence_transformers import SentenceTransformer




# Immutable snapshot state


@dataclass(frozen=True)
class RecommenderState:

    index: hnswlib.Index

    post_ids: list[int]

    post_texts: list[str]

    embeddings: np.ndarray



# Singleton Recommender Service


class RecommenderService:

    _instance = None

    _initialized = False


    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

        return cls._instance


    def __init__(self):

        if self._initialized:
            return


        # Embedding model


        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


        # Current immutable state
        self.state = None


        # Atomic swap lock


        self.swap_lock = threading.Lock()

        self._initialized = True



    # Build HNSW state


    def _build_state(self, posts):

        if len(posts) == 0:
            return RecommenderState(
                index=None,
                post_ids=[],
                post_texts=[],
                embeddings=None
            )

        post_ids = [p['id'] for p in posts]

        texts = [p['text'] for p in posts]

        # Embeddings

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        # HNSW index

        dimension = embeddings.shape[1]

        index = hnswlib.Index(
            space='cosine',
            dim=dimension
        )

        index.init_index(

            max_elements=len(embeddings),

            ef_construction=200,

            M=16
        )

        index.add_items(
            embeddings,
            np.arange(len(embeddings))
        )

        # Search quality/speed tradeoff
        index.set_ef(100)

        # Immutable snapshot

        return RecommenderState(

            index=index,

            post_ids=post_ids,

            post_texts=texts,

            embeddings=embeddings
        )



    # Initial startup


    def initialize(self, posts):

        print("Initializing recommender...")

        state = self._build_state(posts)

        self.state = state

        print("Recommender initialized")



    # Periodic rebuild
    def rebuild(self, posts):
        try:
            print("Rebuilding recommendation index...")

            # Build NEW snapshot in background
            new_state = self._build_state(posts)

            # Atomic swap
            with self.swap_lock:

                self.state = new_state

            print("Recommendation index swapped")
            return {'status': True, 'message': 'Recommendation index swapped'}
        except Exception as e:
            return {'status': False, 'message': str(e)}



    # User embedding
    def _build_user_embedding(self, liked_texts):

        embedding = np.asarray(self.model.encode(

            liked_texts,

            convert_to_numpy=True,

            normalize_embeddings=True

        )).mean(axis=0)

        return np.array(
            [embedding],
            dtype=np.float32
        )


    # Recommendation API
    def recommend(self, liked_texts, liked_post_ids, top_k=50, feed_size=20, explore_probability=0.2):

        # Immutable snapshot
        state = self.state

        if state is None or state.index is None:

            return []

        # User embedding
        user_embedding = self._build_user_embedding(
            liked_texts
        )


        # HNSW ANN search
        indices, distances = state.index.knn_query(user_embedding, k=top_k)

        indices = indices[0]

        distances = distances[0]

        # Candidate pool
        candidates = []

        for idx, distance in zip(indices, distances):

            post_id = state.post_ids[idx]

            if post_id in liked_post_ids:
                continue

            similarity = 1 - distance

            candidates.append(
                (post_id, similarity)
            )

        # Exploration vs exploitation
        feed = []

        used = set()

        all_post_ids = set(state.post_ids)

        random_pool = list(
            all_post_ids - set(liked_post_ids)
        )

        for _ in range(feed_size):
            # Exploration
            if random.random() < explore_probability:

                random.shuffle(random_pool)

                for post_id in random_pool:

                    if post_id not in used:

                        feed.append(post_id)

                        used.add(post_id)

                        break

            # Exploitation
            else:

                for post_id, similarity in candidates:

                    if post_id not in used:

                        feed.append(post_id)

                        used.add(post_id)

                        break

        return feed