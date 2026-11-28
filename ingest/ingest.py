#  Reads each PDF page by page (so every chunk keeps its page number),
#  splits pages into chunks, flags chunks that contain warnings, embeds them, and uploads them:

import glob, os, re, fitz

from backend.common import *


from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)


def make_index():
    """Create the search index, or update it if it already exists."""

    fields = [
        # Unique ID for each chunk. Every index needs exactly one key field.
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),

        # The chunk text. SearchableField means Azure can do keyword search on it.
        SearchableField(name="content", type=SearchFieldDataType.String),

        # Which PDF the chunk came from. filterable lets you filter by manual.
        SimpleField(name="manual", type=SearchFieldDataType.String, filterable=True),

        # The page number. Numbers use SimpleField, not SearchableField,
        # because SearchableField only works for text.
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),

        # True if the chunk mentions a warning, caution or danger.
        SimpleField(name="has_warning", type=SearchFieldDataType.Boolean, filterable=True),

        # The embedding: a list of 1536 decimal numbers, the size that
        # text-embedding-3-small produces. The vector field must use SearchField
        # with searchable=True, and it points to the "vec" profile defined below.
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vec",
        ),
    ]

    # Vector search settings. "algorithms" lists the search methods available
    # (HNSW is a fast way to find similar vectors). "profiles" gives names to
    # the combinations that fields can use.
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vec", algorithm_configuration_name="hnsw")],
    )

    # Semantic search settings. This lets Azure re-rank results by meaning,
    # looking at the "content" field.
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="sem",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    # This is a function CALL: the index definition goes inside the parentheses.
    # If it were written with "=", nothing would be sent to Azure.
    index_client.create_or_update_index(
        SearchIndex(
            name=os.environ["SEARCH_INDEX"],
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
    )
    print("Index is ready:", os.environ["SEARCH_INDEX"])


def chunk_text(text, max_length=1000):
    """Split text into chunks of at most max_length characters,
    breaking at the end of sentences where possible."""

    # Split after ., ! or ? when followed by spaces.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than max_length is cut into pieces,
        # otherwise it would make a chunk that is too big.
        while len(sentence) > max_length:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence[:max_length].strip())
            sentence = sentence[max_length:]

        # If the sentence still fits in the current chunk, add it.
        if len(current) + len(sentence) + 1 <= max_length:
            current += sentence + " "
        else:
            # Otherwise save the current chunk and start a new one.
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "

    # Save the last chunk.
    if current.strip():
        chunks.append(current.strip())

    # Remove empty chunks. Embedding an empty string causes an error.
    return [c for c in chunks if c]


def main():
    # Step 1: make sure the index exists before uploading anything.
    make_index()

    # Step 2: read the PDFs and build a list of chunk documents.
    docs = []

    # Find every PDF in the manuals folder (relative to where you run the script).
    pdf_paths = glob.glob("manuals/*.pdf")
    print(f"Found {len(pdf_paths)} PDF file(s)")

    for path in pdf_paths:
        manual = os.path.basename(path)  # just the file name, e.g. "guide.pdf"

        with fitz.open(path) as pdf:
            # enumerate(..., start=1) numbers the pages from 1, like a human would.
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text()

                for n, chunk in enumerate(chunk_text(page_text)):
                    docs.append({
                        # Keep only letters, digits and dashes: Azure keys only
                        # allow letters, digits, _, - and =.
                        "id": re.sub(r"[^A-Za-z0-9-]", "", f"{manual}-{page_number}-{n}"),
                        "content": chunk,
                        "manual": manual,
                        "page": page_number,
                        # \b means a word boundary, so "warning" matches but
                        # "forewarning" does not. re.I ignores upper/lower case.
                        "has_warning": bool(
                            re.search(r"\bwarning\b|\bcaution\b|\bdanger\b", chunk, re.I)
                        ),
                    })

    print(f"Created {len(docs)} chunks")

    # Step 3: embed and upload in batches of 16, to keep each request small.
    for i in range(0, len(docs), 16):
        batch = docs[i:i + 16]

        # One call to Azure OpenAI gives back one embedding per chunk, in order.
        embeddings = embed([d["content"] for d in batch])

        # zip pairs each document with its matching embedding.
        for doc, vector in zip(batch, embeddings):
            doc["embedding"] = vector

        search_client.upload_documents(batch)
        print(f"Uploaded {len(batch)} documents")

    print("Done. Uploaded", len(docs), "chunks in total")


# This line runs main() only when you start this file directly,
# not when another file imports it.
if __name__ == "__main__":
    main()