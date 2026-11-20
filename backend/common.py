# Shared clients for OpenAI and Azure Search. This is used by both the API and the worker.

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient


load_dotenv()

E = os.environ

oai = AzureOpenAI(azure_endpoint=E['AZURE_OPENAI_ENDPOINT'] )
api_key = E['AZURE_OPENAI_API_KEY']
api_version = E['AZURE_OPENAI_API_VERSION']
cred = AzureKeyCredential(E['SEARCH_kEY'])
search_client = SearchClient(endpoint=E['SEARCH_ENDPOINT'], index_name=E['SEARCH_INDEX'], credential=cred)
index_client = SearchIndexClient(endpoint=E['SEARCH_ENDPOINT'], credential=cred)

def embed(texts):
    out = oai.embeddings.create(model=E['EMBED_DEPLOYMENT'], input=texts)
    return [e.embedding for e in out.data]
