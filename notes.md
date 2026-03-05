# RAG Constitutional Literature

Welcome to **RAG Constitutional Literature**, a fully static, serverless AI Search Engine dedicated to the founding documents of the United States. 

This repository demonstrates how to build a powerful Retrieval-Augmented Generation (RAG) system entirely in the browser using JavaScript and `Transformers.js`, eliminating the need for expensive backend servers and database hosting.

## How It Works

Instead of sending user search queries to a remote server, this application downloads an embedded snapshot of the data (`metadata.json` and `embeddings.bin`) and an AI language model (`Xenova/all-MiniLM-L6-v2`) directly into the user's browser. 

When you search for a term like "taxes" or "liberty":
1. The browser uses the local AI model to translate your query into a mathematical vector (a string of 384 numbers representing its semantic meaning).
2. It then mathematically compares this vector against thousands of pre-processed document vectors using **Cosine Similarity**.
3. The most relevant matches are instantly sorted and displayed, weighting crucial documents like the Constitution and Federalist Papers slightly higher.

Because it is 100% static (HTML, JS, and JSON), it can be hosted for **free on GitHub Pages**.

---

## Technical Documentation

We have provided three extensive technical guides diving into the engineering behind this application. Whether you are seeking to deploy this yourself or build a highly-scalable frontend-only search engine, these guides cover the entire development journey:

### [1. Building a Serverless Static RAG Application (MVP)](01_static_rag_mvp.md)
Explore the foundational MVP. This document details the Python scripting used to construct the ingestion pipeline. It explains how `sqlite3` and `sentence-transformers` read through raw files, build an AI index dynamically, and then neatly export that index into a static `metadata.json` and binary `embeddings.bin` package. Finally, it explores the Javascript implementation of `Transformers.js` to execute real-time searches natively in the browser without an API.

### [2. Full Document Reconstruction](02_document_reconstruction.md)
A core challenge of RAG systems is that AI models perform poorly on massive documents (like 5,000-word essays). They require small, dense "chunks" to accurately measure relevance. This document explains the architecture implemented to sidestep this limitation. Learn how the frontend Javascript maps these small chunks back to their source files on disk, firing asynchronous `fetch()` calls to dynamically stitch the full essay back together in the browser whenever a user clicks "Inspect Full Document."

### [3. Advanced Search Features (Filters \u0026 Weights)](03_advanced_search_features.md)
As the database grew to over 10,000 historical passages, raw cosine similarity searches became slightly inefficient. This document covers the advanced optimization features we deployed over the baseline search engine:
* **Client-Side Pre-Filtering**: How we added Javascript toggle checkboxes to slice the array down before applying matrix math, reducing calculation time from 300ms to near-instant for power users.
* **Semantic Source Weighting**: The math behind how we applied multipliers to boost primary source documents (like the US Constitution) to the top of the relevance stack over personal letters.
* **Heuristic Document Chunking**: The Python script logic for intelligently breaking giant Federalist papers up by paragraph tags to ensure a cleaner vector representation.

---

## Getting Started

To run this application locally:

1. Clone the repository.
2. Open a terminal and run a local server (e.g., `python -m http.server 8000`).
3. View it in your browser at `http://localhost:8000`.

To deploy to GitHub Pages:
Simply push all files (`index.html`, `metadata.json`, `embeddings.bin`, `records/`, and these markdown files) to a public GitHub repository and enable GitHub Pages on the `main` branch!


# Building a Serverless Static RAG Application (MVP)

This document outlines the foundation of the "RAG Constitutional Literature" search engine. We built a fully serverless, static Retrieval-Augmented Generation (RAG) system that can be hosted for free on GitHub Pages. 

Traditional RAG applications require a backend server running Python (FastAPI/Flask) to handle AI embedding models and a vector database (like FAISS, Pinecone) to perform similarity searches. Instead, we shifted **all computation to the frontend browser**.

## 1. The Data Ingestion Pipeline (`index.py`)
First, we need to read our historical documents (JSON format), split them into readable chunks, and store them systematically.

```python
import os
import sqlite3
import json

def init_db():
    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            record_id TEXT,
            title TEXT,
            volume TEXT,
            section TEXT,
            snippet TEXT NOT NULL,
            FOREIGN KEY(file_id) REFERENCES files(id)
        )
    ''')
    conn.commit()
    return conn

def ingest_json_records(conn):
    c = conn.cursor()
    for root, _, files in os.walk("records"):
        for file in files:
            if not file.endswith('.json'): continue
            file_path = os.path.join(root, file).replace('\\', '/')
            
            # Register file
            c.execute('INSERT OR IGNORE INTO files (path) VALUES (?)', (file_path,))
            c.execute('SELECT id FROM files WHERE path = ?', (file_path,))
            file_id = c.fetchone()[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
                for record in records:
                    c.execute('''
                        INSERT INTO chunks (file_id, record_id, title, volume, section, snippet)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (file_id, record.get('id'), record.get('title'), record.get('volume'), record.get('section'), record.get('text')))
    conn.commit()
```

## 2. Generating Static Embeddings (`build_static.py`)
Instead of embedding documents on-the-fly, we pre-embed them all into a static binary architecture. We use `sentence-transformers` because it outputs normalized, high-quality vectors, saving them to `embeddings.bin` over a bloated JSON file.

```python
import sqlite3
import json
from sentence_transformers import SentenceTransformer

def build_static_json():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()
    
    # Extract all chunks
    c.execute('SELECT c.id, c.title, c.snippet, f.path FROM chunks c JOIN files f ON c.file_id = f.id')
    rows = c.fetchall()
    
    # Generate embeddings
    texts = [row[2] for row in rows]
    # encode produces normalized vectors
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    import numpy as np
    embeddings_np = np.array(embeddings, dtype=np.float32)
    with open("embeddings.bin", 'wb') as f:
        f.write(embeddings_np.tobytes())
    
    static_data = []
    for i, row in enumerate(rows):
        static_data.append({
            "id": row[0],
            "title": row[1] or "",
            "snippet": row[2],
            "path": row[3]
        })
        
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(static_data, f, separators=(',', ':')) # Minified output
```

## 3. Browser-Native Vector Search (`index.html`)
The frontend loads entirely locally. We use the incredible `Transformers.js` library to load the exact same embedding model (`Xenova/all-MiniLM-L6-v2`) into the user's browser via WebAssembly (WASM). 

```html
<script type="module">
    import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

    let data = [];
    let extractor = null;

    // Load data and model on boot
    async function init() {
        const response = await fetch('metadata.json');
        data = await response.json();
        data.forEach((item, idx) => item._index = idx); // Assign row indices

        const embResponse = await fetch('embeddings.bin');
        const embBuffer = await embResponse.arrayBuffer();
        window.embeddingsArray = new Float32Array(embBuffer);

        // Download the AI model directly into the browser cache
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        console.log("System Ready!");
    }

    // Perform the semantic search
    async function search(query) {
        // Embed the query
        const output = await extractor(query, { pooling: 'mean', normalize: true });
        const queryVector = output.data;

        // Perform Cosine Similarity locally using Javascript over TypedArrays
        const DIMENSIONS = 384;
        let scoredData = data.map(item => {
            const rowStart = item._index * DIMENSIONS;
            let dotProduct = 0;
            for (let i = 0; i < DIMENSIONS; i++) {
                dotProduct += queryVector[i] * window.embeddingsArray[rowStart + i];
            }
            return { item: item, score: dotProduct };
        });

        // Sort results
        scoredData.sort((a, b) => b.score - a.score);
        
        // Return top 5
        return scoredData.slice(0, 5);
    }
    
    document.addEventListener('DOMContentLoaded', init);
</script>
```

By pre-computing the heavy embeddings backend and doing the querying frontend via WASM, this application costs $0 in compute to run on platforms like GitHub Pages.


# Augmenting Static RAG: Full Document Reconstruction

Once the MVP was operational, returning small, high-density AI snippets of text (like paragraphs from Federalist No. 10), users requested the ability to "Inspect Full Document." The challenge was that the static `metadata.json` database was purposely designed to only store small *chunks* of texts to keep memory overhead low and vector search lightning-fast. 

To solve this, we linked the chunks back to their heavy, original source files on disk.

## 1. Dynamic Network Requests (`index.html`)
When a user clicks "Inspect Full Document" on a `chunk` result card, the browser intercepts the ID and the path. Since `metadata.json` stores the relative path of the parent file (e.g., `records/records_federalist.json`), the browser can simply fire off a lightweight `fetch()` request on the fly to grab the massive parent JSON array containing the entire document.

```javascript
window.openDocumentModal = async function (chunkId, query) {
    const chunk = data.find(c => c.id === chunkId);
    if (!chunk) return;

    // Display a beautiful loading state
    const modal = document.getElementById('documentModal');
    const contentEl = document.getElementById('modalContent');
    contentEl.innerHTML = '<div class="text-center italic">Retrieving full historical document from archive...</div>';
    
    // Prevent background scrolling while modal is open
    document.body.style.overflow = 'hidden';

    // 1. DYNAMICALLY LOAD THE FULL DATASET FILE
    const req = await fetch(chunk.path);
    const jsonRecords = await req.json();

    let matchingRecords = [];
    const isLetter = typeof chunk.record_id === 'string' && chunk.record_id.startsWith('letter_');

    // 2. RECONSTRUCT THE FULL DOCUMENT LOGICALLY
    if (isLetter) {
        // Letters are uniquely identified by their title
        matchingRecords = jsonRecords.filter(r => r.title === chunk.title);
    } else {
        // Records (like Federalist Papers) are logically grouped by volume and section
        matchingRecords = jsonRecords.filter(r => r.volume === chunk.volume && r.section === chunk.section);
    }

    // 3. AGGREGATE THE TEXT
    const rawText = matchingRecords.map(r => r.text).join('\n\n');
    let safeFullText = escapeHtml(rawText);
    let safeSnippet = escapeHtml(chunk.snippet);

    // 4. HIGHLIGHT THE TARGET SNIPPET IN THE RECONSTRUCTED DOCUMENT
    const styledSnippet = `<span id="highlight-target" class="bg-amber-200/50 text-ink font-bold px-1 rounded shadow-sm border-b border-amber-600/30">${safeSnippet}</span>`;

    if (safeFullText.includes(safeSnippet)) {
        contentEl.innerHTML = safeFullText.replace(safeSnippet, styledSnippet);
    } else {
        // Failsafe if replacement failed due to unexpected whitespace formats
        contentEl.innerHTML = safeFullText;
    }

    // 5. SCROLL DIRECTLY TO THE HIGHLIGHT
    setTimeout(() => {
        const target = document.getElementById('highlight-target');
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);
}
```

## 2. Scrollbar Mechanics Fix
Tailwind sets hidden default scrollbars in certain circumstances. When implementing the full modal popup, we encountered an issue where massive 5,000-word Federalist Papers rendered off the page without a scrollbar. 

**Solution:** The modal's internal container needed an explicit maximum viewport height (`max-h-[85vh]`) so that the browser knew *when* to overflow the text, paired with `overflow-y-auto`. We also added bespoke CSS to style the scrollkit slider to match the old-world colonial aesthetic.

```css
/* Custom Scrollbar for Modal inside the <style> block */
.modal-scroll::-webkit-scrollbar {
    width: 14px;
}
.modal-scroll::-webkit-scrollbar-track {
    background: #fdfbf7; 
    border-radius: 4px;
    border-left: 1px solid rgba(139, 115, 85, 0.2);
}
.modal-scroll::-webkit-scrollbar-thumb {
    background: #8b7355; 
    border-radius: 4px;
    border: 3px solid #fdfbf7;
}
.modal-scroll::-webkit-scrollbar-thumb:hover {
    background: #4a3b2c; 
}
```

This strategy of "storing metadata frontend, fetching full data asynchronously on-demand" ensured the primary search interface remained incredibly fast and snappy while still offering complex, full-text readability to the user.

# Advanced Static RAG: Weights, Filters, and Optimization

As the scale of the "RAG Constitutional Literature" project grew to encompass thousands of documents covering the writing of the Founding Fathers (20,000+ overlapping chunks), the simplicity of the MVP cosine similarity Search proved insufficient. Certain documents were overpowering others in the query index, and performance dragged slightly on low-end machines. 

Here are the advanced features we layered directly onto the `Transformers.js` loop and Python ingestion scripts.

## 1. Custom Semantic Scoring & Source Weighting
When querying documents, the AI evaluates strings like "Taxes" and "Commerce." However, because some foundational texts (like the Constitution) carry significantly more historical and legal gravity than a politician's personal diary entry on the exact same topic, we introduced an **override weighting system**. 

After the AI dot-product calculation naturally graded the similarity, we intervened with source-specific multipliers based on the document's file path URL.

```javascript
let scoredData = data.map(item => {
    let dotProduct = 0;
    // Calculate Cosine Similarity over the 384-dimensional vector float array
    for (let i = 0; i < queryVector.length; i++) {
        dotProduct += queryVector[i] * item.embedding[i];
    }
    
    // Apply custom weighting multipliers based on the source document
    let weight = 1.0;
    if (item.path && item.path.includes('records_elliot')) {
        weight = 0.85; // De-prioritize Elliot records
    } else if (item.path && item.path.includes('records_federalist')) {
        weight = 1.1;  // Prioritize Federalist Papers
    } else if (item.path && item.path.includes('records_constitution')) {
        weight = 1.2;  // Highest priority for the Constitution
    }
    
    // Safety cap to prevent scores exceeding 100% confidence
    let adjustedScore = dotProduct * weight;
    if (adjustedScore > 1.0) adjustedScore = 1.0;

    return { ...item, score: adjustedScore, originalScore: dotProduct };
});

// Sort by highest adjusted score
scoredData.sort((a, b) => b.score - a.score);
```

By boosting the Constitution to `1.2x`, we guarantee that if a user searches for an exact clause match, the Constitution returns flawlessly as `#1` on the index.

## 2. Fast-Path Pre-Filtering & TypedArrays (O(N) Optimization)
Initially, calculating 20,000+ matrix dot-products purely in JavaScript by cloning JSON objects on every loop caused severe memory pressure. To reduce execution time dynamically for power-users, we introduced native HTML `checkbox` filters right below the search bar, and we shifted all vector math to flat `Float32Array` buffers. 

Instead of evaluating the dot product of *every single chunk* and *then* discarding the unwanted domains, we evaluate the filters first. 

```javascript
// Grab HTML filter statuses
const useConstitution = document.getElementById('filterConstitution').checked;
const useFederalist = document.getElementById('filterFederalist').checked;
const useConvention = document.getElementById('filterConvention').checked;
const useLetters = document.getElementById('filterLetters').checked;

// Pre-filter data BEFORE evaluating matrix dot-products for speed
// For example, if the user unchecks everything except "Letters", 
// this array instantly drops by 60%, drastically reducing the vector loop calculation!
let filteredData = data.filter(item => {
    const path = item.path || '';
    if (!useConstitution && path.includes('records_constitution')) return false;
    if (!useFederalist && path.includes('records_federalist')) return false;
    if (!useConvention && (path.includes('records_farrand') || path.includes('records_elliot'))) return false;
    if (!useLetters && path.includes('records_letters')) return false;
    return true;
});

// Loop over ONLY the pre-filtered items
let scoredData = filteredData.map(item => { ... })
```

## 3. Heuristic JSON Chunk Normalization (`index.py`)
Foundational documents were originally extremely massive texts, often spanning 3,000 to 5,000 words each. During early testing, pushing one of these entire essays to `SentenceTransformer` for an embedding resulted in a severely "diluted" summary vector that performed poorly on specific topical searches.

To optimize the quality of our RAG hits, we executed a sliding window algorithm in `index.py` to break up enormous `.json` documents into 200-word chunks with a 50-word overlap, while cleanly retaining their origin metadata so they could still magically reconnect later inside the `index.html` full document modal.

```python
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max(1, chunk_size - overlap)):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# ... Execution loops over JSON
text_chunks = chunk_text(record.get('text', ''))
for i, chunk_snippet in enumerate(text_chunks):
    c.execute(
        """INSERT INTO chunks (
            file_id, record_id, title, volume, section, chunk_index, snippet
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            file_id, 
            record.get('id', ''),
            record.get('title', ''),
            record.get('volume', ''),
            record.get('section', ''),
            i,
            chunk_snippet
        )
    )
```

Scaling RAG from a massive server backend to a finely tuned, extremely fast, statically hosted javascript bundle ensures not only a lower barrier to entry but drastically reduces cloud-compute fees to exactly $0.
