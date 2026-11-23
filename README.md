# PDF RAG application

A small Python FastAPI application for indexing and querying the supplied
`Wuthering Heights` PDF.

## Setup

```powershell
py -m pip install -r requirement.txt
py -m pip install pypdf
py -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the web interface.

## API

- `GET /health`
- `POST /query` with JSON `{"question": "Who is Heathcliff?", "top_k": 5}`
- `POST /index/upload` with a PDF as the `file` form field

The default PDF is indexed automatically at startup.
