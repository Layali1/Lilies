import streamlit as st
from dotenv import load_dotenv

from core.ingest import get_store, add_pdf
from core.graph import build_graph
from core.utils import cite

load_dotenv(override=True)

st.set_page_config(
    page_title="Lilies",
    page_icon="🌸",
    layout="centered"
)

# --------------------------------------------------
# 1. Lily design
# --------------------------------------------------

LILY = """<svg viewBox="-50 -50 100 100" width="{w}" xmlns="http://www.w3.org/2000/svg">
{petals}
<g stroke="#B0457A" stroke-width="1.2" stroke-linecap="round">
<line x1="0" y1="0" x2="0" y2="-14"/>
<line x1="0" y1="0" x2="12" y2="-8"/>
<line x1="0" y1="0" x2="-12" y2="-8"/>
</g>
<circle cx="0" cy="-14" r="2.2" fill="#E8A33D"/>
<circle cx="12" cy="-8" r="2.2" fill="#E8A33D"/>
<circle cx="-12" cy="-8" r="2.2" fill="#E8A33D"/>
</svg>"""

PETALS = "".join(
    f'<ellipse cx="0" cy="-26" rx="9" ry="24" '
    f'transform="rotate({a})" fill="#F7C6DC" '
    f'stroke="#E79AC0" stroke-width="1"/>'
    for a in range(0, 360, 60)
)

lily = lambda w: LILY.format(w=w, petals=PETALS)


# --------------------------------------------------
# 2. Styling
# --------------------------------------------------

st.markdown(f"""
<style>

.stApp {{
    background: linear-gradient(
        180deg,
        #FFF8FB 0%,
        #FDEAF3 100%
    );
}}

h1 {{
    font-weight: 600;
    letter-spacing: .5px;
}}

[data-testid="stChatMessage"] {{
    background: #FFFFFFCC;
    border: 1px solid #F5CFE0;
    border-radius: 18px;
}}

[data-testid="stChatMessage"] p {{
    unicode-bidi: plaintext;
    text-align: start;
}}

[data-testid="stSidebar"] {{
    border-right: 1px solid #F5CFE0;
}}

.lily-corner {{
    position: fixed;
    pointer-events: none;
    opacity: .55;
    z-index: 0;
}}

</style>

<div class="lily-corner"
     style="right:-30px;bottom:-20px;transform:rotate(-15deg)">
    {lily(220)}
</div>

<div class="lily-corner"
     style="left:-40px;top:60px;transform:rotate(20deg);opacity:.35">
    {lily(160)}
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. App title
# --------------------------------------------------

st.title("Lilies")

st.caption(
    "Upload PDFs and ask about them in Arabic or English"
)


# --------------------------------------------------
# 4. ChromaDB
# --------------------------------------------------

@st.cache_resource
def store():
    return get_store()


# --------------------------------------------------
# 5. Session state
# --------------------------------------------------

if "chats" not in st.session_state:

    st.session_state.chats = {
        "chat_1": {
            "title": "New Chat",
            "msgs": []
        }
    }

    st.session_state.active_chat = "chat_1"
    st.session_state.chat_counter = 1


if "done" not in st.session_state:
    st.session_state.done = set()


# --------------------------------------------------
# 6. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown(lily(90), unsafe_allow_html=True)

    # Upload PDFs
    st.markdown("### Your PDFs")

    files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    for f in files or []:

        if f.name not in st.session_state.done:

            bar = st.progress(
                0.0,
                text=f"Processing {f.name}..."
            )

            n = add_pdf(
                store(),
                f,
                f.name,
                progress=lambda p: bar.progress(
                    p,
                    text=f"{f.name}: {int(p * 100)}%"
                )
            )

            bar.empty()

            st.session_state.done.add(f.name)

            if n:
                st.success(f"{f.name}: {n} chunks added")
            else:
                st.warning(
                    f"{f.name}: No extractable text found"
                )

    st.divider()

    # New chat
    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_counter += 1

        chat_id = f"chat_{st.session_state.chat_counter}"

        st.session_state.chats[chat_id] = {
            "title": "New Chat",
            "msgs": []
        }

        st.session_state.active_chat = chat_id

        st.rerun()

    # Conversation history
    st.markdown("### Chats")

    for chat_id, chat in st.session_state.chats.items():

        label = chat["title"]

        if chat_id == st.session_state.active_chat:
            label = f"✦ {label}"

        if st.button(
            label,
            key=f"select_{chat_id}",
            use_container_width=True
        ):

            st.session_state.active_chat = chat_id

            st.rerun()


# --------------------------------------------------
# 7. Current conversation
# --------------------------------------------------

active_chat = st.session_state.active_chat

msgs = st.session_state.chats[active_chat]["msgs"]


# --------------------------------------------------
# 8. Build RAG graph
# --------------------------------------------------

if "graph" not in st.session_state:

    st.session_state.graph = build_graph(store())


# --------------------------------------------------
# 9. Display conversation history
# --------------------------------------------------

for m in msgs:

    with st.chat_message(m["role"]):

        st.markdown(m["text"])

        if m["role"] == "assistant" and m.get("sources"):

            with st.expander("Sources"):

                for source in m["sources"]:

                    st.markdown(
                        f"**{source['filename']}** · "
                        f"{source['citation']}"
                    )

                    st.markdown(source["content"])


# --------------------------------------------------
# 10. Ask a question
# --------------------------------------------------

if q := st.chat_input("اسأل عن ملفاتك"):

    # Save user message
    msgs.append({
        "role": "user",
        "text": q
    })

    # Give conversation a title based on first question
    if st.session_state.chats[active_chat]["title"] == "New Chat":

        st.session_state.chats[active_chat]["title"] = q[:35]

    with st.chat_message("user"):

        st.markdown(q)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            out = st.session_state.graph.invoke({
                "question": q
            })

        st.markdown(out["answer"])

        sources = []

        if out.get("relevant"):

            with st.expander("Sources"):

                for d in out["docs"]:

                    source = {
                        "filename": d.metadata["source"],
                        "citation": cite(d.metadata),
                        "content": d.page_content[:300]
                    }

                    sources.append(source)

                    st.markdown(
                        f"**{source['filename']}** · "
                        f"{source['citation']}"
                    )

                    st.markdown(source["content"])

    # Save assistant response
    msgs.append({
        "role": "assistant",
        "text": out["answer"],
        "sources": sources
    })