"""
OrgDNA MVP — Enterprise Memory Platform demo
--------------------------------------------
Two workspaces:
  1. Northeastern ITS — real public knowledge base (10 docs)
  2. NU Decision Archive (simulated) — mock Teams threads, emails, meeting
     minutes, and decision records in the university's voice, demonstrating
     how OrgDNA unifies memory across the tools an organization already uses.

Pipeline: TF-IDF retrieval (title+section enriched) -> optional live
site:northeastern.edu fallback (serper.dev, ITS workspace only) ->
grounded answer composed by Groq Llama 3.3 (free) with per-channel citations.
"""

import os
import re
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).parent
WORKSPACES = {
    "Northeastern ITS": {
        "dir": APP_DIR / "knowledge_base",
        "desc": "Real public IT Services knowledge",
        "web_fallback": True,
        "samples": [
            "My laptop broke — can I borrow one?",
            "I can't connect to university Wi-Fi",
            "How do I set up the VPN?",
            "I got a phishing email — what do I do?",
        ],
    },
    "NU Decision Archive (simulated)": {
        "dir": APP_DIR / "nu_memory",
        "desc": "Simulated Teams, email, meetings & decision records in the university's voice",
        "web_fallback": False,
        "samples": [
            "Why did we standardize on GlobalProtect for VPN?",
            "Who was involved in the VPN decision?",
            "Why are laptop loans five days?",
            "Where do I find how our processes work?",
        ],
    },
}
CHANNEL_ICONS = {"teams": "💬 Teams", "email": "✉️ Email",
                 "meeting": "🎙️ Meeting", "decision": "📋 Decision Archive",
                 "doc": "📚 Document"}
SERPER_THRESHOLD = 0.15
GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="OrgDNA — Enterprise Memory MVP",
                   page_icon="🧬", layout="centered",
                   initial_sidebar_state="expanded")

# ------------------------------ Styling ---------------------------------- #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero {
    background: linear-gradient(135deg, #1E2761 0%, #3D5AA9 100%);
    border-radius: 18px; padding: 2.2rem 2.4rem 2rem 2.4rem;
    color: white; margin-bottom: 1.2rem;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .4rem 0; color: white; }
.hero p  { color: #CADCFC; margin: 0; font-size: 1rem; line-height: 1.5; }
.hero .badge {
    display: inline-block; background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
    padding: .2rem .8rem; font-size: .75rem; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: .8rem; color: #CADCFC;
}
.stats { display: flex; gap: .8rem; margin: 0 0 1.4rem 0; }
.stat { flex: 1; background: #F3F6FD; border: 1px solid #E3E9F7;
        border-radius: 14px; padding: .9rem 1rem; text-align: center; }
.stat .n { font-size: 1.35rem; font-weight: 800; color: #1E2761; }
.stat .l { font-size: .72rem; color: #5A6270; text-transform: uppercase; letter-spacing: .05em; }
div[data-testid="stHorizontalBlock"] button {
    border-radius: 999px !important; border: 1px solid #D5DEF2 !important;
    background: #FFFFFF !important; color: #1E2761 !important;
    font-size: .82rem !important; padding: .35rem .9rem !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #3D5AA9 !important; background: #F3F6FD !important;
}
[data-testid="stChatMessage"] {
    border-radius: 16px; border: 1px solid #EDF1FA; background: #FFFFFF;
    box-shadow: 0 1px 4px rgba(30,39,97,.05);
    padding: .4rem .6rem; margin-bottom: .5rem;
}
.src-badge {
    display: inline-block; border-radius: 999px; padding: .12rem .65rem;
    font-size: .72rem; font-weight: 600; margin-bottom: .35rem; margin-right: .3rem;
}
.src-local { background: #E8EEFC; color: #1E2761; }
.src-web   { background: #E6F6F0; color: #0B6E4F; }
.src-ai    { background: #FDF1E7; color: #9A5B13; }
section[data-testid="stSidebar"] { background: #F7F9FE; }
section[data-testid="stSidebar"] .kb-item {
    font-size: .8rem; color: #3A4256; padding: .28rem .6rem;
    background: white; border: 1px solid #E7ECF8;
    border-radius: 10px; margin-bottom: .3rem;
}
.pill-on  { color: #0B6E4F; font-weight: 700; }
.pill-off { color: #B03A2E; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


def get_secret(name: str) -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, "")


def channel_of(filename: str) -> str:
    for prefix in ("teams", "email", "meeting", "decision"):
        if filename.startswith(prefix):
            return prefix
    return "doc"


# ----------------------------- Indexing ---------------------------------- #

@st.cache_resource(show_spinner="Indexing organizational memory...")
def build_index(kb_dir_str: str):
    """Chunk the KB. Each chunk is indexed WITH its doc title and section
    header so questions match even when phrased differently than the text."""
    kb_dir = Path(kb_dir_str)
    display, indexed, sources, titles, channels = [], [], [], [], []
    for f in sorted(kb_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        m = re.search(r"^#\s*(.+)$", text, re.MULTILINE)
        title = m.group(1).strip() if m else f.stem
        ch = channel_of(f.name)
        section = ""
        for p in re.split(r"\n\s*\n", text):
            p = p.strip()
            if not p:
                continue
            if p.startswith("##"):
                section = p.lstrip("# ").strip()
                continue
            if p.startswith("#"):
                continue
            display.append(p)
            indexed.append(f"{title}. {section}. {p}")
            sources.append(f.name)
            titles.append(title)
            channels.append(ch)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return vec, vec.fit_transform(indexed), display, sources, titles, channels


def retrieve(kb_dir: Path, question: str, k: int = 6):
    vec, matrix, chunks, sources, titles, channels = build_index(str(kb_dir))
    sims = cosine_similarity(vec.transform([question]), matrix).flatten()
    order = sims.argsort()[::-1][:k]
    return [{"chunk": chunks[i], "source": sources[i], "title": titles[i],
             "score": float(sims[i]), "kind": "local", "channel": channels[i]}
            for i in order if sims[i] > 0.02]


# --------------------------- Serper live search -------------------------- #

def serper_search(question: str, num: int = 3):
    key = get_secret("SERPER_API_KEY")
    if not key:
        return []
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": f"{question} site:northeastern.edu", "num": num},
            timeout=10,
        )
        r.raise_for_status()
        return [{"chunk": it.get("snippet", ""), "source": it.get("link", ""),
                 "title": it.get("title", "Web result"), "score": None,
                 "kind": "web", "channel": "web"}
                for it in r.json().get("organic", [])[:num] if it.get("snippet")]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_full_paragraphs(url: str, question: str, max_paras: int = 3) -> str:
    """Fetch the page behind a web result and return the FULL paragraphs
    most relevant to the question — instead of Google's trimmed snippet."""
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0 (OrgDNA-MVP)"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
            tag.decompose()
        paras = list(dict.fromkeys(
            re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            for el in soup.find_all(["p", "li"])
            if 60 <= len(el.get_text(strip=True)) <= 900))
        if not paras:
            return ""
        vec = TfidfVectorizer(stop_words="english")
        m = vec.fit_transform(paras + [question])
        sims = cosine_similarity(m[-1], m[:-1]).flatten()
        top = sorted(sims.argsort()[::-1][:max_paras])
        return "\n\n".join(paras[i] for i in top if sims[i] > 0.05)
    except Exception:
        return ""


# ------------------------- Groq answer generation ------------------------- #

def groq_answer(question: str, hits: list, history: list, workspace: str) -> str | None:
    """Compose a grounded answer with Groq's free Llama 3.3 70B."""
    key = get_secret("GROQ_API_KEY")
    if not key:
        return None
    context = "\n\n---\n\n".join(
        f"[SOURCE TYPE: {CHANNEL_ICONS.get(h['channel'], h['channel'])} | "
        f"{h['title']} | {h['source']}]\n{h['chunk']}"
        for h in hits
    )
    recent = [m for m in history[-6:] if m["role"] == "user"]
    convo = "\n".join(f"- {m['content']}" for m in recent[:-1]) or "(none)"
    if "Decision Archive" in workspace:
        persona = ("You are OrgDNA, an organizational memory assistant for "
                   "Northeastern University staff. You answer questions about "
                   "university decisions and processes using memory captured "
                   "from Teams, email, meetings, and the decision archive "
                   "(simulated for this demo). When answering about a "
                   "decision, cover what was decided, why, who was involved, "
                   "where it happened, and the outcome — when the context "
                   "provides them — and mention which channels the answer "
                   "came from.")
        fallback = ("If the context does not contain the answer, say the "
                    "organizational memory doesn't cover it yet.")
    else:
        persona = ("You are OrgDNA, an organizational memory assistant "
                   "answering questions about Northeastern University IT "
                   "Services.")
        fallback = ("If the context does not contain the answer, say so and "
                    "direct the user to the IT Service Desk: 617-373-4357 or "
                    "help@northeastern.edu (24/7).")
    system = (
        f"{persona}\nRules:\n"
        "1. Answer ONLY from the provided context. Never invent names, "
        "dates, numbers, URLs, or policies.\n"
        "2. Be direct: give the answer first, then supporting detail in "
        "short paragraphs or a brief list.\n"
        "3. Synthesize across passages when several are relevant.\n"
        f"4. {fallback}\n"
        "5. Keep answers under ~170 words. No preamble."
    )
    user = (f"Earlier questions in this conversation:\n{convo}\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}")
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.2, "max_tokens": 550},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# ------------------------------ Sidebar ----------------------------------- #

with st.sidebar:
    st.markdown("## 🧬 OrgDNA")
    st.caption("Enterprise Memory Platform · MVP")
    ws_name = st.radio("**Workspace**", list(WORKSPACES.keys()),
                       captions=[w["desc"] for w in WORKSPACES.values()])
    ws = WORKSPACES[ws_name]
    if not ws["dir"].exists() or not list(ws["dir"].glob("*.md")):
        st.error(f"No documents found in `{ws['dir'].name}/`. "
                 "Make sure the folder sits next to app.py and contains .md files.")
    if "Decision Archive" in ws_name:
        st.info("This workspace simulates memory ingested from Teams, "
                "email, meetings, and a decision archive — the connector "
                "pipeline the full product automates. Records are "
                "illustrative, not real university data.", icon="🔌")
    st.divider()
    serper_on = bool(get_secret("SERPER_API_KEY")) and ws["web_fallback"]
    groq_on = bool(get_secret("GROQ_API_KEY"))
    st.markdown(f"**AI answers (Groq):** "
                f"<span class='{'pill-on' if groq_on else 'pill-off'}'>"
                f"{'● ON' if groq_on else '● OFF — extractive mode'}</span>",
                unsafe_allow_html=True)
    st.markdown(f"**Live web fallback:** "
                f"<span class='{'pill-on' if serper_on else 'pill-off'}'>"
                f"{'● ON' if serper_on else '● OFF'}</span>",
                unsafe_allow_html=True)
    always_web = st.toggle("Always add live web context", value=False,
                           disabled=not serper_on)
    st.divider()
    files = sorted(ws["dir"].glob("*.md"))
    st.markdown(f"**📚 Memory sources** ({len(files)} items)")
    for f in files:
        first = f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        icon = CHANNEL_ICONS[channel_of(f.name)].split()[0]
        st.markdown(f"<div class='kb-item'>{icon} {first}</div>",
                    unsafe_allow_html=True)
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------- Header ----------------------------------- #

hero_sub = ("Ask your organization anything. This demo remembers Northeastern's "
            "IT Services knowledge — laptops, Wi-Fi, VPN, phishing, and more."
            if "Northeastern ITS" == ws_name else
            "Ask the university's memory. OrgDNA remembers decisions, Teams "
            "threads, emails, and meetings — and answers with who, why, where, "
            "and outcome. (Simulated records.)")
st.markdown(f"""
<div class="hero">
  <div class="badge">MVP · {ws_name}</div>
  <h1>🧬 OrgDNA</h1>
  <p>{hero_sub}</p>
</div>
""", unsafe_allow_html=True)

_, _, chunks_all, _, _, channels_all = build_index(str(ws["dir"]))
n_channels = len(set(channels_all))
n_msgs = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
st.markdown(f"""
<div class="stats">
  <div class="stat"><div class="n">{len(files)}</div><div class="l">Sources</div></div>
  <div class="stat"><div class="n">{n_channels}</div><div class="l">Channels</div></div>
  <div class="stat"><div class="n">{len(chunks_all)}</div><div class="l">Memory chunks</div></div>
  <div class="stat"><div class="n">{n_msgs}</div><div class="l">Questions asked</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("**Try one of these, or ask your own:**")
cols = st.columns(len(ws["samples"]))
clicked = None
for c, sample in zip(cols, ws["samples"]):
    if c.button(sample, use_container_width=True):
        clicked = sample

if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("last_ws") != ws_name:      # new workspace = new chat
    st.session_state.messages = []
    st.session_state.last_ws = ws_name

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧬" if m["role"] == "assistant" else "🙋"):
        st.markdown(m["content"], unsafe_allow_html=True)

question = st.chat_input("Ask your organization's memory...") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)

    hits = retrieve(ws["dir"], question)
    best_local = hits[0]["score"] if hits else 0.0
    local_is_good = best_local >= SERPER_THRESHOLD
    used_web = False

    if serper_on and (always_web or not local_is_good):
        with st.spinner("Searching northeastern.edu live..."):
            web_hits = serper_search(question)
        if web_hits:
            full = fetch_full_paragraphs(web_hits[0]["source"], question)
            if full:
                web_hits[0]["chunk"] = full
            hits += web_hits
            used_web = True

    with st.chat_message("assistant", avatar="🧬"):
        if not hits:
            answer = ("I couldn't find that in this workspace's memory. "
                      + ("The IT Service Desk can help 24/7: **617-373-4357** "
                         "or **help@northeastern.edu**."
                         if "Northeastern ITS" == ws_name else
                         "It may predate what's been ingested so far."))
            st.markdown(answer)
        else:
            ai_text = None
            if groq_on:
                with st.spinner("Composing answer from organizational memory..."):
                    ai_text = groq_answer(question, hits,
                                          st.session_state.messages, ws_name)
            used_channels = {h["channel"] for h in hits[:4] if h["kind"] == "local"}
            badges = "".join(
                f"<span class='src-badge src-local'>{CHANNEL_ICONS[c]}</span>"
                for c in sorted(used_channels))
            if used_web:
                badges += "<span class='src-badge src-web'>🌐 Live web</span>"
            if ai_text:
                badges += "<span class='src-badge src-ai'>✨ AI-composed</span>"
                answer = f"{badges}<br><br>{ai_text}"
            else:
                # extractive fallback (no Groq key or API error)
                best = hits[0] if (local_is_good or not used_web) else \
                       next((h for h in hits if h["kind"] == "web"), hits[0])
                answer = f"{badges}<br><br>{best['chunk']}"
                if best["kind"] == "web":
                    answer += (f"<br><br><small>Source: "
                               f"<a href='{best['source']}'>{best['source']}</a></small>")
            st.markdown(answer, unsafe_allow_html=True)

            with st.expander(f"🔍 Sources — {len(hits)}"
                             + (" (incl. live web)" if used_web else "")):
                for h in hits:
                    if h["kind"] == "web":
                        st.markdown(f"**🌐 {h['title']}**  \n{h['source']}")
                    else:
                        st.markdown(f"**{CHANNEL_ICONS[h['channel']]} — "
                                    f"{h['title']}** · `{h['source']}` · "
                                    f"similarity **{h['score']:.2f}**")
                    st.markdown("> " + h["chunk"].replace("\n", "\n> "))
                    st.markdown("---")
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("<br><div style='text-align:center;color:#9AA3B5;font-size:.78rem'>"
            "OrgDNA MVP · unified memory across documents, Teams, email & "
            "meetings · TF-IDF + Groq Llama 3.3 · BA&IE final project</div>",
            unsafe_allow_html=True)