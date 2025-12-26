# PDF → langextract → Embeddings + FAISS Retrieval (NLQ)

This repository contains a **three-stage document processing pipeline** implemented as Jupyter notebooks.  
The pipeline processes **PDF documents**, extracts structured information using `langextract`, and enables **embedding-based retrieval and natural-language querying (NLQ)** using sentence embeddings and FAISS.

The notebooks are designed to be run **sequentially**, with each stage building on the outputs of the previous one.

---

## Repository Structure

notebooks/
* 01_ingest_parse.ipynb
* 02_langextract_extract.ipynb
* 03_embed_retrieve_nlq.ipynb


Each notebook represents a distinct processing step and can be reviewed independently.

---

## Notebook 1 — PDF Ingestion & Parsing  
**`01_ingest_parse.ipynb`**

### What this notebook does
- Reads PDF files using `pypdf.PdfReader`
- Iterates through PDF pages and extracts raw text
- Cleans extracted text using regex-based preprocessing
- Structures extracted content into tabular form using `pandas`

### Key components
- `extract_pdf_pages()`  
  Reads PDFs page-by-page and returns extracted text.
- `clean_text()`  
  Normalizes whitespace and removes unwanted characters.
- Uses:
  - `pypdf`
  - `pandas`
  - `pathlib`
  - `re`
  - `tqdm`

### Output
- Cleaned, page-level text suitable for downstream extraction.
- Structured data representation for later processing.

---

## Notebook 2 — Structured Extraction with `langextract`  
**`02_langextract_extract.ipynb`**

### What this notebook does
- Loads cleaned text produced in the ingestion step
- Uses the `langextract` library to extract structured information from text
- Manages configuration via environment variables using `python-dotenv`
- Formats and inspects extraction results using `pandas`

### Key components
- Imports `langextract as lx`
- Uses `load_dotenv()` to load environment variables
- Handles JSON-formatted extraction outputs
- Uses:
  - `langextract`
  - `python-dotenv`
  - `pandas`
  - `json`
  - `textwrap`

### Output
- Structured extraction results serialized in JSON-like form
- Tabular views of extracted fields for inspection and validation

> Note: This notebook performs **extraction only**. It does not perform embedding, retrieval, or question answering.

---

## Notebook 3 — Embedding, Retrieval & NLQ  
**`03_embed_retrieve_nlq.ipynb`**

### What this notebook does
- Loads extracted text from the previous step
- Splits text into chunks for embedding
- Generates vector embeddings using `SentenceTransformer`
- Indexes embeddings using FAISS for similarity search
- Implements retrieval and NLQ helper functions

### Key components
- `chunk_text()`  
  Splits text into fixed-size chunks for embedding.
- `retrieve()`  
  Performs similarity search over FAISS index.
- `load_extractions()`  
  Loads extracted content for embedding.
- `nlq()`  
  Executes embedding-based retrieval for natural-language queries.

### Technologies used
- `sentence-transformers` (`SentenceTransformer`)
- `faiss`
- `numpy`
- `pandas`

### Output
- FAISS similarity search results
- Ranked text chunks relevant to a user query

> Note: This notebook performs **retrieval only**. It does not generate answers or perform LLM-based reasoning.

---

## Execution Order

Run the notebooks in the following order:

1. `01_ingest_parse.ipynb`
2. `02_langextract_extract.ipynb`
3. `03_embed_retrieve_nlq.ipynb`

Each step depends on artifacts created by the previous stage.

---

## Dependencies

Based strictly on notebook imports:

```txt
pandas
numpy
tqdm
pypdf
langextract
python-dotenv
sentence-transformers
faiss-cpu
## Scope & Limitations

### What this repository does
- Parses PDFs  
- Cleans and structures text  
- Extracts structured information using `langextract`  
- Enables embedding-based similarity retrieval and NLQ  

### What this repository does not do
- No LLM answer generation  
- No RAG pipeline  
- No agent orchestration  
- No model fine-tuning or training  

---

## Intended Use

This project demonstrates a practical document processing workflow commonly used in analytics and applied NLP contexts, including:
- Document ingestion pipelines  
- Structured text extraction  
- Embedding-based search and retrieval  

The implementation prioritizes clarity, modularity, and inspectability over abstraction.
