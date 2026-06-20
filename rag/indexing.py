import ast

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    PointStruct,
    SparseVector,
    PayloadSchemaType,
)
from config import COLLECTION, SAMPLE_SIZE, DENSE_DIM


class RAGIndexer:
    def __init__(self, model, csv_path):

        self.model = model
        self.csv_path = csv_path

    def preprocess_dataframe(self):
        df = pd.read_csv(self.csv_path)
        df = df.sample(SAMPLE_SIZE, random_state=42)
        df["description"] = df["description"].fillna("")

        def convert_to_str(row: pd.Series) -> str:
            ingredients = ast.literal_eval(row["ingredients"])
            embed_text = (
                f"Name: {row['name']}. "
                f"Ingredients: {', '.join(ingredients)}. "
                f"Description: {row['description']}."
            )
            return embed_text

        df["embed_text"] = df.apply(convert_to_str, axis=1)
        return df

    def embed_batch(self, texts: list[str], batch_size: int = 12):
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            batch_size=batch_size,
        )
        dense = out["dense_vecs"]  # ndarray (N, 1024)
        sparse = out["lexical_weights"]  # list[defaultdict]
        return dense, sparse

    @staticmethod
    def to_qdrant_sparse(weights: dict):
        return {
            "indices": [int(k) for k in weights.keys()],
            "values": [float(v) for v in weights.values()],
        }

    def build_payload(self, row: pd.Series) -> dict:
        nutrition = ast.literal_eval(row["nutrition"])
        nutrition_fields = [
            "calories",
            "fat",
            "sugar",
            "sodium",
            "protein",
            "sat_fat",
            "carbs",
        ]
        return {
            "name": row["name"],
            "minutes": int(row["minutes"]),
            "ingredients": ast.literal_eval(row["ingredients"]),
            "tags": ast.literal_eval(row["tags"]),
            "steps": ast.literal_eval(row["steps"]),
            "n_ingredients": int(row["n_ingredients"]),
            "calories": nutrition[0],
            "nutrition": dict(zip(nutrition_fields, nutrition)),
        }

    def create_collection(self, client: QdrantClient):
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        client.create_payload_index(COLLECTION, "tags", PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "minutes", PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION, "calories", PayloadSchemaType.FLOAT)

    def upsert_recipes(self, client, ids, dense, sparse, payloads, batch_size=128):
        points = []
        for i in range(len(ids)):
            s = self.to_qdrant_sparse(sparse[i])
            points.append(
                PointStruct(
                    id=int(ids[i]),
                    vector={
                        "dense": dense[i].tolist(),
                        "sparse": SparseVector(
                            indices=s["indices"], values=s["values"]
                        ),
                    },
                    payload=payloads[i],
                )
            )
        for j in range(0, len(points), batch_size):
            client.upsert(COLLECTION, points=points[j : j + batch_size])
        return len(points)

    def run(self, client: QdrantClient) -> int:
        df = self.preprocess_dataframe()
        dense, sparse = self.embed_batch(df["embed_text"].tolist())
        payloads = df.apply(self.build_payload, axis=1).tolist()
        ids = df["id"].tolist()
        self.create_collection(client)
        return self.upsert_recipes(client, ids, dense, sparse, payloads)
