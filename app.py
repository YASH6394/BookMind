from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "book-emily-bronte-wuthering-heights.pdf"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    page: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


def tokenize(text: str) -> list[str]:
    return [match.lower() for match in TOKEN_PATTERN.findall(text)]


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, re.sub(r"\s+", " ", text).strip()))
    if not pages:
        raise HTTPException(status_code=400, detail="The PDF contains no extractable text.")
    return pages


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for page_number, text in pages:
        words = text.split()
        if not words:
            continue
        for start in range(0, len(words), chunk_words - overlap_words):
            content = " ".join(words[start : start + chunk_words]).strip()
            if content:
                chunks.append({"page": page_number, "text": content})
            if start + chunk_words >= len(words):
                break
    return chunks


class RAGIndex:
    def __init__(self) -> None:
        self.chunks: list[dict[str, object]] = []
        self.document_frequency: dict[str, int] = {}
        self.document_vectors: list[dict[str, float]] = []
        self.document_norms: list[float] = []

    def build(self, pages: Iterable[tuple[int, str]]) -> int:
        self.chunks = chunk_pages(pages)
        if not self.chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created.")

        self.document_frequency = {}
        term_vectors: list[dict[str, float]] = []
        for chunk in self.chunks:
            frequencies: dict[str, int] = {}
            for token in tokenize(str(chunk["text"])):
                frequencies[token] = frequencies.get(token, 0) + 1
            for token in frequencies:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1
            term_vectors.append({token: 1.0 + math.log(count) for token, count in frequencies.items()})

        document_count = len(term_vectors)
        self.document_vectors = []
        self.document_norms = []
        for vector in term_vectors:
            weighted = {}
            for token, value in vector.items():
                inverse_document_frequency = math.log(
                    (document_count + 1) / (self.document_frequency[token] + 1)
                ) + 1.0
                weighted[token] = value * inverse_document_frequency
            norm = math.sqrt(sum(value * value for value in weighted.values()))
            self.document_vectors.append(weighted)
            self.document_norms.append(norm)
        return len(self.chunks)

    def search(self, question: str, top_k: int) -> list[dict[str, object]]:
        query_frequencies: dict[str, int] = {}
        for token in tokenize(question):
            query_frequencies[token] = query_frequencies.get(token, 0) + 1

        document_count = len(self.document_vectors)
        query_vector: dict[str, float] = {}
        for token, count in query_frequencies.items():
            if token not in self.document_frequency:
                continue
            inverse_document_frequency = math.log(
                (document_count + 1) / (self.document_frequency[token] + 1)
            ) + 1.0
            query_vector[token] = (1.0 + math.log(count)) * inverse_document_frequency

        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0:
            return []

        ranked: list[tuple[float, int]] = []
        for index, document_vector in enumerate(self.document_vectors):
            dot_product = sum(
                value * document_vector.get(token, 0.0)
                for token, value in query_vector.items()
            )
            denominator = query_norm * self.document_norms[index]
            score = dot_product / denominator if denominator else 0.0
            ranked.append((score, index))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self.chunks[index] for score, index in ranked[:top_k] if score > 0]


index = RAGIndex()
app = FastAPI(title="Wuthering Heights PDF RAG")


if DEFAULT_PDF.exists():
    index.build(extract_pdf_pages(DEFAULT_PDF))


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "indexed_chunks": len(index.chunks)}


@app.post("/index/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    temporary_path = BASE_DIR / f"_upload_{file.filename}"
    try:
        temporary_path.write_bytes(await file.read())
        chunk_count = index.build(extract_pdf_pages(temporary_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {"filename": file.filename, "indexed_chunks": chunk_count}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not index.chunks:
        raise HTTPException(status_code=409, detail="Index a PDF before querying.")

    chunks = index.search(request.question, request.top_k)
    if not chunks:
        return QueryResponse(
            answer="I could not find relevant text for that question in the indexed PDF.",
            sources=[],
        )

    sources = [
        Source(page=int(chunk["page"]), text=str(chunk["text"])[:1200])
        for chunk in chunks
    ]
    combined = " ".join(str(chunk["text"]) for chunk in chunks)
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(combined) if len(sentence.strip()) >= 30]
    question_terms = set(tokenize(request.question))
    ranked_sentences = sorted(
        sentences,
        key=lambda sentence: sum(token in set(tokenize(sentence)) for token in question_terms),
        reverse=True,
    )
    answer = " ".join(ranked_sentences[:3])
    if not answer:
        answer = str(chunks[0]["text"])[:1500]
    return QueryResponse(answer=answer, sources=sources)


@app.get("/", response_class=HTMLResponse)
def interface() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>BookMind - PDF RAG</title>
        <style>
          :root {
            --ink: #172033;
            --muted: #687083;
            --purple: #7056d9;
            --purple-dark: #5842b6;
            --surface: #ffffff;
            --line: #e8e9ef;
            --soft: #f5f3fb;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            min-height: 100vh;
            background: #eef0f5;
            color: var(--ink);
            font-family: Inter, "Segoe UI", Arial, sans-serif;
          }
          .shell {
            width: min(1180px, calc(100% - 32px));
            height: calc(100vh - 32px);
            margin: 16px auto;
            display: grid;
            grid-template-columns: 280px 1fr;
            overflow: hidden;
            background: var(--surface);
            border: 1px solid rgba(23, 32, 51, .06);
            border-radius: 20px;
            box-shadow: 0 18px 60px rgba(35, 43, 70, .12);
          }
          aside {
            display: flex;
            flex-direction: column;
            padding: 24px 20px 18px;
            background: #fbfbfd;
            border-right: 1px solid var(--line);
          }
          .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -.5px;
          }
          .brand-mark {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 11px;
            color: white;
            background: linear-gradient(135deg, #8069e8, #5b42b9);
          }
          .new-chat {
            width: 100%;
            margin: 30px 0 18px;
            padding: 12px 14px;
            border: 0;
            border-radius: 10px;
            background: var(--soft);
            color: var(--purple-dark);
            font-weight: 700;
            cursor: pointer;
          }
          .section-label {
            margin: 8px 6px 10px;
            color: #a0a4b1;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.2px;
            text-transform: uppercase;
          }
          .document-card {
            display: flex;
            align-items: center;
            gap: 11px;
            padding: 11px;
            border-radius: 12px;
            background: white;
            border: 1px solid var(--line);
          }
          .pdf-icon {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            flex: 0 0 auto;
            border-radius: 9px;
            background: #fff0f0;
            color: #e45a5a;
            font-size: 10px;
            font-weight: 900;
          }
          .document-name {
            min-width: 0;
            font-size: 13px;
            font-weight: 750;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .document-status {
            margin-top: 3px;
            color: #57a076;
            font-size: 11px;
          }
          .upload-label {
            display: block;
            margin-top: 10px;
            padding: 10px;
            border: 1px dashed #cbd0dd;
            border-radius: 10px;
            color: var(--muted);
            text-align: center;
            font-size: 12px;
            cursor: pointer;
          }
          .profile {
            display: flex;
            align-items: center;
            gap: 9px;
            margin-top: auto;
            padding: 12px 6px 0;
            color: var(--muted);
            font-size: 12px;
          }
          .avatar {
            width: 32px;
            height: 32px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: #e6dff8;
            color: var(--purple-dark);
            font-weight: 800;
          }
          main {
            min-width: 0;
            display: grid;
            grid-template-rows: auto 1fr auto;
          }
          header {
            height: 78px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 34px;
            border-bottom: 1px solid var(--line);
          }
          .title { font-size: 16px; font-weight: 800; }
          .health {
            display: flex;
            align-items: center;
            gap: 7px;
            color: #5c9c78;
            font-size: 12px;
          }
          .health-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #62bd86;
            box-shadow: 0 0 0 4px #edf8f1;
          }
          #chat {
            overflow-y: auto;
            padding: 28px 34px;
            scroll-behavior: smooth;
          }
          .welcome {
            max-width: 620px;
            margin: 28px auto 32px;
            text-align: center;
          }
          .welcome h1 {
            margin: 0 0 8px;
            font-size: clamp(28px, 4vw, 42px);
            letter-spacing: -1.5px;
          }
          .welcome h1 span { color: var(--purple); }
          .welcome p {
            margin: 0 auto 24px;
            color: var(--muted);
            line-height: 1.6;
          }
          .suggestions {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
          }
          .suggestion {
            padding: 13px 14px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: white;
            color: var(--ink);
            text-align: left;
            cursor: pointer;
            transition: .18s ease;
          }
          .suggestion:hover {
            border-color: #bcb1ed;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(112, 86, 217, .08);
          }
          .message {
            max-width: 760px;
            margin: 16px 0;
          }
          .message.user {
            margin-left: auto;
            width: fit-content;
            max-width: 72%;
            padding: 12px 16px;
            border-radius: 16px 16px 4px 16px;
            background: var(--purple);
            color: white;
          }
          .answer-block {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 10px;
          }
          .bot-avatar {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: var(--soft);
            color: var(--purple);
            font-weight: 900;
          }
          .answer {
            line-height: 1.7;
            white-space: pre-wrap;
          }
          .sources {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
          }
          .source {
            max-width: 100%;
            border: 0;
            border-radius: 8px;
            padding: 8px 10px;
            background: #f4f5f8;
            color: #686d7b;
            font-size: 12px;
            cursor: pointer;
          }
          .source-text {
            display: none;
            width: 100%;
            margin-top: 8px;
            padding: 12px;
            border-left: 3px solid var(--purple);
            background: #faf9fe;
            color: var(--muted);
            line-height: 1.55;
          }
          .composer-wrap {
            padding: 14px 34px 22px;
          }
          .composer {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: end;
            gap: 10px;
            min-height: 58px;
            padding: 8px 8px 8px 18px;
            border: 1px solid #dddfea;
            border-radius: 16px;
            background: white;
            box-shadow: 0 8px 28px rgba(35, 43, 70, .07);
          }
          #question {
            width: 100%;
            min-height: 40px;
            resize: none;
            border: 0;
            outline: 0;
            color: var(--ink);
            font: 14px/20px "Segoe UI", Arial, sans-serif;
          }
          #send {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border: 0;
            border-radius: 11px;
            background: var(--purple);
            color: white;
            cursor: pointer;
          }
          #send:disabled { opacity: .55; cursor: wait; }
          .typing {
            display: inline-flex;
            gap: 4px;
            padding: 10px 0;
          }
          .typing i {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #aaa4bd;
            animation: bounce 1s infinite alternate;
          }
          .typing i:nth-child(2) { animation-delay: .2s; }
          .typing i:nth-child(3) { animation-delay: .4s; }
          @keyframes bounce { to { transform: translateY(-5px); } }
          @media (max-width: 760px) {
            .shell { width: 100%; height: 100vh; margin: 0; grid-template-columns: 1fr; border-radius: 0; }
            aside { display: none; }
            header { height: 64px; padding: 0 18px; }
            #chat { padding: 18px; }
            .welcome { margin-top: 12px; }
            .suggestions { grid-template-columns: 1fr; }
            .composer-wrap { padding: 10px 14px 14px; }
          }
        </style>
      </head>
      <body>
        <div class="shell">
          <aside>
            <div class="brand"><div class="brand-mark">B</div> BookMind</div>
            <button class="new-chat" onclick="resetChat()">+ New conversation</button>
            <div class="section-label">Library</div>
            <div class="document-card">
              <div class="pdf-icon">PDF</div>
              <div style="min-width:0">
                <div class="document-name" id="document-name">Wuthering Heights</div>
                <div class="document-status" id="document-status">Indexed</div>
              </div>
            </div>
            <label class="upload-label" for="pdf-upload">Upload another PDF</label>
            <input id="pdf-upload" type="file" accept=".pdf,application/pdf" hidden>
            <div class="profile"><div class="avatar">U</div><span>Personal workspace</span></div>
          </aside>
          <main>
            <header>
              <div class="title">Document assistant</div>
              <div class="health"><span class="health-dot"></span><span id="chunk-count">Ready</span></div>
            </header>
            <section id="chat">
              <div class="welcome" id="welcome">
                <h1>Ask about your <span>book</span></h1>
                <p>Explore characters, themes, and key moments from Wuthering Heights.</p>
                <div class="suggestions">
                  <button class="suggestion" onclick="useSuggestion(this)">Who is Heathcliff?</button>
                  <button class="suggestion" onclick="useSuggestion(this)">What is the main theme?</button>
                  <button class="suggestion" onclick="useSuggestion(this)">Summarize the story</button>
                  <button class="suggestion" onclick="useSuggestion(this)">Who is Catherine Earnshaw?</button>
                </div>
              </div>
            </section>
            <div class="composer-wrap">
              <form class="composer" id="composer">
                <textarea id="question" rows="1" placeholder="Ask a question about the document..."></textarea>
                <button id="send" type="submit" aria-label="Send">➤</button>
              </form>
            </div>
          </main>
        </div>
        <script>
          const chat = document.getElementById('chat');
          const question = document.getElementById('question');
          const send = document.getElementById('send');

          async function loadHealth() {
            try {
              const response = await fetch('/health');
              const data = await response.json();
              document.getElementById('chunk-count').textContent =
                data.indexed_chunks + ' sections indexed';
            } catch (_) {}
          }

          function useSuggestion(button) {
            question.value = button.textContent;
            question.focus();
          }

          function resetChat() {
            chat.innerHTML = `
              <div class="welcome" id="welcome">
                <h1>Ask about your <span>book</span></h1>
                <p>Explore characters, themes, and key moments from Wuthering Heights.</p>
                <div class="suggestions">
                  <button class="suggestion" onclick="useSuggestion(this)">Who is Heathcliff?</button>
                  <button class="suggestion" onclick="useSuggestion(this)">What is the main theme?</button>
                  <button class="suggestion" onclick="useSuggestion(this)">Summarize the story</button>
                  <button class="suggestion" onclick="useSuggestion(this)">Who is Catherine Earnshaw?</button>
                </div>
              </div>`;
          }

          function appendUser(text) {
            const node = document.createElement('div');
            node.className = 'message user';
            node.textContent = text;
            chat.appendChild(node);
          }

          function appendTyping() {
            const node = document.createElement('div');
            node.className = 'message answer-block';
            node.id = 'typing';
            node.innerHTML =
              '<div class="bot-avatar">B</div><div class="typing"><i></i><i></i><i></i></div>';
            chat.appendChild(node);
            chat.scrollTop = chat.scrollHeight;
          }

          function appendAnswer(data) {
            document.getElementById('typing')?.remove();
            const node = document.createElement('div');
            node.className = 'message answer-block';
            const sources = (data.sources || []).map((source, index) => `
              <div>
                <button class="source" onclick="toggleSource(this)">Page ${source.page}</button>
                <div class="source-text">${escapeHtml(source.text)}</div>
              </div>`).join('');
            node.innerHTML =
              '<div class="bot-avatar">B</div><div>' +
              '<div class="answer"></div>' +
              (sources ? '<div class="sources">' + sources + '</div>' : '') +
              '</div>';
            node.querySelector('.answer').textContent = data.answer;
            chat.appendChild(node);
            chat.scrollTop = chat.scrollHeight;
          }

          function escapeHtml(value) {
            const div = document.createElement('div');
            div.textContent = value;
            return div.innerHTML;
          }

          function toggleSource(button) {
            const text = button.nextElementSibling;
            text.style.display = text.style.display === 'block' ? 'none' : 'block';
          }

          async function ask() {
            const text = question.value.trim();
            if (!text) return;
            document.getElementById('welcome')?.remove();
            appendUser(text);
            question.value = '';
            question.style.height = '40px';
            send.disabled = true;
            appendTyping();
            try {
              const response = await fetch('/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: text, top_k: 5})
              });
              const data = await response.json();
              if (!response.ok) throw new Error(data.detail || 'Request failed');
              appendAnswer(data);
            } catch (error) {
              appendAnswer({answer: error.message, sources: []});
            } finally {
              send.disabled = false;
              question.focus();
            }
          }

          document.getElementById('composer').onsubmit = event => {
            event.preventDefault();
            ask();
          };

          question.addEventListener('input', () => {
            question.style.height = '40px';
            question.style.height = Math.min(question.scrollHeight, 120) + 'px';
          });
          question.addEventListener('keydown', event => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              ask();
            }
          });

          document.getElementById('pdf-upload').addEventListener('change', async event => {
            const file = event.target.files[0];
            if (!file) return;
            const form = new FormData();
            form.append('file', file);
            document.getElementById('document-status').textContent = 'Indexing...';
            try {
              const response = await fetch('/index/upload', {method: 'POST', body: form});
              const data = await response.json();
              if (!response.ok) throw new Error(data.detail || 'Upload failed');
              document.getElementById('document-name').textContent = file.name;
              document.getElementById('document-status').textContent = 'Indexed';
              loadHealth();
              resetChat();
            } catch (error) {
              document.getElementById('document-status').textContent = error.message;
            }
            event.target.value = '';
          });

          loadHealth();
        </script>
      </body>
    </html>
    """
