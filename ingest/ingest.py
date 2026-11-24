#  Reads each PDF page by page (so every chunk keeps its page number),
#  splits pages into chunks, flags chunks that contain warnings, embeds them, and uploads them:

import glob, os, re, fitz
from backend.common import *
from azure.search.documents.indexes.models import (SearchIndex, SimpleField,
SearchableField, SearchField,
SearchFieldDataType, VectorSearch, HnswAlgorithmConfiguration,
VectorSearchProfile,
SemanticConfiguration, SemanticPrioritizedFields, SemanticField,
SemanticSearch)


def make_index():
    # Create a new index with vector search and semantic search enabled
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="manual", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="has_warning", type=SearchFieldDataType.Boolean, filterable=True),
        SearchableField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=1536, vector_search_profile_name="vec"),
    ]

    vs = VectorSearch(
        algorithm_configurations=[
            HnswAlgorithmConfiguration(name="hnsw")], profile=VectorSearchProfile(name="vec", vector_search_algorithm_configuration_name="hnsw"))

    sem = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="sem",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                )
            )])

    index_client.create_or_update_index = SearchIndex(
        name=os.environ['SEARCH_INDEX'],
        fields=fields,
        vector_search=vs,
        semantic_search=sem)


def chunk_text(text, max_length=1000):
    # Split the text into chunks of max_length characters, trying to split at sentence boundaries
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():

    # Create or prepare the search index
    make_index()

    # Empty list to store all document chunks
    docs = []

    # Find all PDF files inside the manuals folder
    for path in glob.glob("manuals/*.pdf"):

        # Get just the PDF filename
        manual = os.path.basename(path)

        # Open the PDF and read each page
        with fitz.open(path) as pdf:

            for pno, page in enumerate(pdf, start=1):

                # Extract text from the current page
                page_text = page.get_text()

                # Split the page text into smaller chunks
                for n, c in enumerate(chunk_text(page_text)):

                    # Create a document with its information
                    docs.append({

                        # Create a unique ID for each chunk
                        "id": re.sub(
                            r"[^A-Za-z0-9-]",
                            "",
                            f"{manual}-{pno}-{n}"
                        ),

                        # Actual text content
                        "content": c,

                        # Original PDF filename
                        "manual": manual,

                        # Original PDF page number
                        "page": pno,

                        # Check whether the chunk contains safety words
                        "has_warning": bool(
                            re.search(
                                r"\bwarning\b|\bcaution\b|\bdanger\b",
                                c,
                                re.I
                            )
                        )
                    })


    # Process 16 chunks at a time
    for i in range(0, len(docs), 16):

        # Get the next 16 documents
        batch = docs[i:i + 16]

        # Extract text from each document and generate embeddings
        embeddings = embed([d["content"] for d in batch])

        # Add each embedding to its matching document
        for d, v in zip(batch, embeddings):

            d["embedding"] = v


        search_client.upload_documents(batch)

        print(f"Uploaded {len(batch)} documents")

    # Display total number of chunks processed
    print("Uploaded", len(docs), "chunks")




main()