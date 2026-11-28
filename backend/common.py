# Shared clients for OpenAI and Azure Search. This is used by both the API and the worker.

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient


load_dotenv()

E = os.environ

oai = AzureOpenAI(
    azure_endpoint=E['AZURE_OPENAI_ENDPOINT'],  # https://<resource>.openai.azure.com/
    api_version=E['AZURE_OPENAI_API_VERSION'],  # e.g. 2024-10-21
    api_key=E['AZURE_OPENAI_API_KEY'],          # plain string
)

cred = AzureKeyCredential(E['SEARCH_KEY'])      # Search wants the wrapper
search_client = SearchClient(endpoint=E['SEARCH_ENDPOINT'], index_name=E['SEARCH_INDEX'], credential=cred)
index_client = SearchIndexClient(endpoint=E['SEARCH_ENDPOINT'], credential=cred)

EMBED_MODEL = E['EMBED_DEPLOYMENT']  #  =deployment name


def embed(texts):
    out = oai.embeddings.create(input=texts, model=EMBED_MODEL)
    return [e.embedding for e in out.data]