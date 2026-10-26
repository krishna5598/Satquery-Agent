import faiss
import numpy as np


class FAISSIndex:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.metadata = []

    def add(self, embeddings, metadata):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=4):

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            item = dict(
                self.metadata[index]
            )

            item["score"] = float(score)

            results.append(item)

        return results