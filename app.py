import os
import glob
import json
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv

import boto3

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None


# -----------------------------
# Config
# -----------------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
CHAT_MODEL_ID = os.getenv("BEDROCK_CHAT_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


# -----------------------------
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, source: str, chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:

    text = clean_text(text)
    chunks: List[Chunk] = []
    if not text:
        return chunks

    start = 0
    cid = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, source=source, chunk_id=cid))
            cid += 1
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def load_resume_and_jobs():
    resume = None
    jobs = {}

    resume_path = os.path.join(DATA_DIR, "resume.txt")
    if os.path.exists(resume_path):
        with open(resume_path, "r", encoding="utf-8", errors="ignore") as f:
            resume = ("resume.txt", f.read())

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "job*.txt"))):
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            jobs[name] = f.read()

    return resume, jobs


def get_bedrock_runtime():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def titan_embed_text(brt, texts: List[str]) -> np.ndarray:
    """
    Calls Titan embeddings model for each text.
    Returns np array of shape (len(texts), dim).
    """
    vectors = []
    for t in texts:
        body = json.dumps({"inputText": t})
        resp = brt.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        emb = payload.get("embedding")
        if emb is None:
            raise RuntimeError(f"No embedding returned. Response keys: {list(payload.keys())}")
        vectors.append(np.array(emb, dtype=np.float32))
    return np.vstack(vectors)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    # Normalize to unit vectors for cosine similarity via inner product
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError(
            "faiss is not available. Install faiss-cpu successfully, then rerun."
        )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if embeddings are normalized
    index.add(embeddings)
    return index


def retrieve_top_k(
    brt,
    index,
    chunks: List[Chunk],
    query: str,
    k: int = 6,
) -> List[Tuple[Chunk, float]]:
    q_emb = titan_embed_text(brt, [query])
    q_emb = normalize_rows(q_emb)

    scores, ids = index.search(q_emb, k)
    results = []
    for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((chunks[idx], float(score)))
    return results


def call_claude_with_context(brt, question: str, contexts: List[Chunk]) -> str:
    """
    Claude 3 Messages API format for Bedrock (Anthropic).
    We enforce citations by asking it to cite [source#chunk] tags.
    """
    # Build context block with citation tags
    context_lines = []
    for c in contexts:
        tag = f"[{c.source}#{c.chunk_id}]"
        context_lines.append(f"{tag}\n{c.text}")
    context_block = "\n\n---\n\n".join(context_lines)

    system_prompt = (
        "You are a personal career assistant. Use ONLY the provided context to make claims about the user or the job.\n"
        "If the context is missing info, say what is missing.\n"
        "When you use facts, include citations using the tags exactly as provided, like [resume.txt#0] or [job1.txt#2].\n"
        "Keep the writing clear and practical.\n"
    )

    user_content = (
        "CONTEXT (resume + job postings):\n"
        f"{context_block}\n\n"
        "TASK:\n"
        f"{question}\n\n"
        "OUTPUT RULES:\n"
        "- Be direct and useful.\n"
        "- If writing resume bullets, make them copy-ready.\n"
        "- If suggesting improvements, prioritize the most important ones first.\n"
        "- Cite factual claims.\n"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 900,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_content}
        ],
    }

    resp = brt.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())

    # Claude output format typically: {"content":[{"type":"text","text":"..."}], ...}
    content = payload.get("content", [])
    if isinstance(content, list) and content:
        # join all text parts
        texts = []
        for part in content:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts).strip()

    # fallback
    return json.dumps(payload, indent=2)

def build_index_for_selection(resume_doc, selected_job_name, jobs_dict, chunk_size, overlap):
    docs = []

    if resume_doc is None:
        raise RuntimeError("resume.txt not found in ./data/")
    docs.append(resume_doc)

    if selected_job_name not in jobs_dict:
        raise RuntimeError(f"{selected_job_name} not found.")
    docs.append((selected_job_name, jobs_dict[selected_job_name]))

    all_chunks: List[Chunk] = []
    for name, text in docs:
        all_chunks.extend(chunk_text(text, source=name, chunk_size=chunk_size, overlap=overlap))

    if not all_chunks:
        raise RuntimeError("No chunks were created from the selected documents.")

    brt = get_bedrock_runtime()
    texts = [c.text for c in all_chunks]
    embs = titan_embed_text(brt, texts)
    embs = normalize_rows(embs)
    index = build_faiss_index(embs)

    return index, all_chunks

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Career RAG Bot", layout="wide")
st.title("Career RAG Bot (Amazon Bedrock + FAISS)")
st.caption("Chat with your resume and a selected job posting, grounded with citations.")

resume_doc, jobs_dict = load_resume_and_jobs()

with st.sidebar:
    st.header("Settings")
    st.write(f"Region: **{AWS_REGION}**")
    st.write(f"Embed model: **{EMBED_MODEL_ID}**")
    st.write(f"Chat model: **{CHAT_MODEL_ID}**")

    chunk_size = st.slider("Chunk size (chars)", 400, 1400, 900, 50)
    overlap = st.slider("Overlap (chars)", 0, 400, 150, 25)
    top_k = st.slider("Top-K retrieval", 2, 12, 6, 1)

    st.divider()
    st.subheader("Select target job")
    if not jobs_dict:
        st.error("No job*.txt files found in ./data/")
        st.stop()

    selected_job = st.selectbox("Job posting", list(jobs_dict.keys()))
    rebuild = st.button("Rebuild index")

    st.divider()
    st.subheader("Quick actions")
    action_missing = st.button("Top Missing Keywords")
    action_summary = st.button("Tailored Resume Summary")
    action_bullets = st.button("Rewrite 3 Project Bullets")
    action_cover = st.button("Cover Letter Draft")
    action_interview = st.button("Interview Questions")

# Rebuild index when selected job changes or user clicks rebuild
selection_key = f"{selected_job}_{chunk_size}_{overlap}"
if "selection_key" not in st.session_state:
    st.session_state.selection_key = None

if rebuild or st.session_state.selection_key != selection_key:
    with st.spinner("Building embeddings + FAISS index for selected job..."):
        try:
            index, chunks = build_index_for_selection(
                resume_doc,
                selected_job,
                jobs_dict,
                chunk_size,
                overlap,
            )
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.selection_key = selection_key
            st.success(f"Index ready for {selected_job} ✅")
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            st.stop()

# Build default prompt from quick actions
preset_prompt = ""

if action_missing:
    preset_prompt = (
        f"Compare my resume against {selected_job}. "
        f"List the most important missing keywords, tools, and qualifications. "
        f"Group them into 'Must-have gaps' and 'Nice-to-have gaps'."
    )
elif action_summary:
    preset_prompt = (
        f"Write a tailored professional resume summary for me based on my resume and {selected_job}. "
        f"Keep it concise, strong, and ready to paste into a resume. Use only supported facts."
    )
elif action_bullets:
    preset_prompt = (
        f"Based on my resume and {selected_job}, rewrite 3 of my strongest project bullets so they align better with the job. "
        f"Make them resume-ready, action-oriented, and ATS-friendly."
    )
elif action_cover:
    preset_prompt = (
        f"Write a tailored cover letter draft for {selected_job} using my resume. "
        f"Keep it professional, natural, and specific to my background. "
        f"Do not invent experience I do not have."
    )
elif action_interview:
    preset_prompt = (
        f"Generate 10 likely interview questions for {selected_job} based on my resume. "
        f"For each question, briefly explain why it is likely and what part of my background it targets."
    )

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Ask")
    question = st.text_area(
        f"Ask about resume.txt + {selected_job}",
        value=preset_prompt,
        height=160,
    )

    ask = st.button("Generate Answer")

    if ask:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            brt = get_bedrock_runtime()

            with st.spinner("Retrieving relevant chunks..."):
                results = retrieve_top_k(
                    brt=brt,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    query=question.strip(),
                    k=top_k,
                )

            contexts = [c for c, _ in results]

            with st.spinner("Calling Bedrock chat model..."):
                try:
                    answer = call_claude_with_context(brt, question.strip(), contexts)
                except Exception as e:
                    st.error(f"Model call failed: {e}")
                    st.stop()

            st.markdown("### Answer")
            st.write(answer)

with col2:
    st.subheader("Top retrieved citations")
    st.write(f"Showing snippets used for **resume.txt + {selected_job}**")

    if ask and "results" in locals():
        for (chunk, score) in results:
            st.markdown(f"**[{chunk.source}#{chunk.chunk_id}]** (score: {score:.3f})")
            with st.expander("Show snippet"):
                st.write(chunk.text)
            st.divider()
    else:
        st.info("Generate an answer to see retrieved chunks here.")

st.divider()
st.caption("Tip: Add more files like job3.txt, job4.txt, etc. into the data folder to compare against more roles.")