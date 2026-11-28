from backend.common import *
from azure.search.documents.models import VectorizedQuery


FIELDS = ["content", "manual", "page", "has_warning"]

def retrieve(question, mode="rerank", k=5, only_warnings=False):
    vq = VectorizedQuery(
        vector=embed([question])[0],
        k_nearest_neighbors=50,
        fields="embedding")
    args = dict(top=k, select = FIELDS)

    if only_warnings:
        args["filter"] = "has_warning eq true"

    if mode == "vector":
        results = search_client.search(search_text=None, vector_queries=[vq], **args)
    elif mode == "hybrid":
        results = search_client.search(search_text=question, vector_queries=[vq], **args)
    else:
        results = search_client.search(search_text=question, vector_queries=[vq], query_type="semantic",semantic_configuration_name="sem", **args)

    return [dict(r) for r in results]