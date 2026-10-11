import html

import streamlit as st

from api import call, err

PAGES: dict = {}   # filled by app.py so pages can switch to each other

MODES = {"Auto": "auto", "My notes only": "material", "General knowledge": "general"}
MODE_HELP = ("Auto: use my uploaded notes when they help, otherwise answer from general knowledge "
             "(clearly labelled). My notes only: strict, refuses if my notes don't cover it. "
             "General knowledge: ignore my notes.")
SUGGESTIONS = ["Summarize my uploaded notes",
               "What are the most important ideas in my notes?",
               "Explain machine learning to a beginner"]


# ------------------------------------------------------------------ look & feel
def inject_css():
    st.markdown("""
<style>
.stDeployButton, #MainMenu, footer {display:none !important;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.lm-hero {background: linear-gradient(135deg,#6C5CE7 0%,#0FA3B1 100%); padding: 26px 32px;
          border-radius: 18px; color: #fff; margin-bottom: 1.2rem;}
.lm-hero h1 {margin:0; font-size:2rem; color:#fff; padding:0;}
.lm-hero p {margin:.35rem 0 0; opacity:.92; font-size:1.05rem;}
div[data-testid="stChatMessage"] {border-radius: 14px; padding: .7rem 1rem; margin-bottom: .4rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
div[data-testid="stMetric"] {background: rgba(124,108,240,.10); border-radius: 14px; padding: 12px 16px;}
.stButton > button {border-radius: 10px;}
</style>""", unsafe_allow_html=True)


def hero(title: str, subtitle: str = ""):
    st.markdown(f'<div class="lm-hero"><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def new_chat():
    st.session_state["conv_id"] = None
    st.session_state["msgs"] = []


def render_msg(m: dict):
    with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        for c in m.get("citations") or []:
            with st.expander(f"📄 [{c['ref']}] {c['filename']} · page {c['page_number']} · match {c['similarity']}"):
                st.write(c["text"])


# ------------------------------------------------------------------ login
def login_page():
    if n := st.session_state.pop("notice", None):
        st.info(n)
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        hero("🎓 LearnMate", "Your adaptive AI tutor. It learns what you know and what to study next.")
        tab_in, tab_up = st.tabs(["Log in", "Create account"])
        with tab_in:
            with st.form("login"):
                email = st.text_input("Email")
                pw = st.text_input("Password", type="password")
                go = st.form_submit_button("Log in", type="primary", width="stretch")
            if go:
                r = call("POST", "/auth/login", data={"username": email, "password": pw})
                if r.ok:
                    st.session_state["token"] = r.json()["access_token"]
                    new_chat()
                    st.rerun()
                else:
                    st.error(err(r))
        with tab_up:
            with st.form("register"):
                name = st.text_input("Your name")
                email2 = st.text_input("Email ")
                pw2 = st.text_input("Password (8-72 characters)", type="password")
                go2 = st.form_submit_button("Create account", width="stretch")
            if go2:
                r = call("POST", "/auth/register",
                         json={"email": email2, "password": pw2, "display_name": name or None})
                if r.ok:
                    st.success("Account created. Switch to the 'Log in' tab.")
                else:
                    st.error(err(r))


# ------------------------------------------------------------------ sidebar (below the page links)
def sidebar():
    ss = st.session_state
    me = ss.get("me", {})
    with st.sidebar:
        st.divider()
        st.text(f"👤 {me.get('display_name') or me.get('email', '')}")
        if st.button("➕ New chat", width="stretch", type="primary", key="sb_new"):
            new_chat()
            st.switch_page(PAGES["chat"])
        # History is saved on the server but stays hidden until the user opens this section.
        with st.expander("🕘 Chat history", expanded=False):
            r = call("GET", "/conversations", params={"limit": 20})
            convs = r.json() if r.ok else []
            if not convs:
                st.caption("No saved chats yet.")
            for c in convs:
                is_open = c["id"] == ss.get("conv_id")
                label = ("▸ " if is_open else "") + c["title"]
                if st.button(label[:34], key=f"conv_{c['id']}", width="stretch",
                             type="primary" if is_open else "secondary"):
                    ss["conv_id"], ss["msgs"] = c["id"], None
                    st.switch_page(PAGES["chat"])
        st.divider()
        if st.button("Log out", width="stretch", key="sb_logout"):
            ss.clear()
            st.rerun()
        with st.expander("Danger zone"):
            sure = st.checkbox("Permanently delete my account and all my data")
            if st.button("Delete my account", disabled=not sure, key="sb_delacct"):
                call("DELETE", "/auth/me")
                ss.clear()
                st.rerun()


# ------------------------------------------------------------------ Home (welcome screen)
def home():
    r = call("GET", "/profile")
    p = r.json() if r.ok else None
    name = (p or {}).get("user", {}).get("display_name") or "there"
    hero(f"Welcome back, {name} 👋", "What would you like to learn today?")
    if p:
        a, b, c = st.columns(3)
        a.metric("Subjects", p["subject_count"])
        b.metric("Documents ready", p["ready_document_count"])
        c.metric("Study passages", p["chunk_count"])
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 💬 Ask LearnMate")
            st.caption("Ask about your notes, or about anything else. Answers are explained simply.")
            if st.button("Start a new chat", type="primary", width="stretch", key="h_chat"):
                new_chat()
                st.switch_page(PAGES["chat"])
    with c2:
        with st.container(border=True):
            st.markdown("#### 📚 Add study material")
            st.caption("Upload PDFs or text files and organise them by subject.")
            if st.button("Open library", width="stretch", key="h_lib"):
                st.switch_page(PAGES["library"])
    with c3:
        with st.container(border=True):
            st.markdown("#### 🧩 Concept map")
            st.caption("See the key concepts in your material and what depends on what.")
            if st.button("Open concept map", width="stretch", key="h_cm"):
                st.switch_page(PAGES["concepts"])
    st.caption("Your past chats are saved. Open **🕘 Chat history** in the sidebar whenever you want them.")


# ------------------------------------------------------------------ Chat
def chat():
    ss = st.session_state
    name = ss.get("me", {}).get("display_name") or "there"
    ss.setdefault("conv_id", None)
    if ss.get("msgs") is None:      # load a saved conversation only when the user opened it
        ss["msgs"] = []
        if ss["conv_id"]:
            r = call("GET", f"/conversations/{ss['conv_id']}/messages")
            if r.ok:
                ss["msgs"] = r.json()
            else:
                ss["conv_id"] = None

    subjects = call("GET", "/subjects").json()
    labels = {s["name"]: s["id"] for s in subjects}
    options = ["All material", *labels]
    if ss.get("chat_subject") not in options:
        ss.pop("chat_subject", None)

    c1, c2, c3 = st.columns([2, 3.2, 1])
    with c1:
        subj = st.selectbox("Subject", options, key="chat_subject")
    with c2:
        mode_label = st.radio("Answer from", list(MODES), horizontal=True, key="chat_mode", help=MODE_HELP)
    with c3:
        if ss["conv_id"]:
            st.write("")
            with st.popover("🗑 Delete"):
                st.write("Delete this chat permanently?")
                if st.button("Yes, delete", key="del_conv"):
                    call("DELETE", f"/conversations/{ss['conv_id']}")
                    new_chat()
                    st.rerun()

    q = st.chat_input("Ask LearnMate anything...") or ss.pop("pending_q", None)

    if not ss["msgs"] and not q:
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(f"Hi **{html.escape(name)}**! 👋 I'm LearnMate. Ask me anything. I'll use your "
                        "uploaded notes when they help, and otherwise explain from general knowledge. "
                        "What would you like to learn today?")
        cols = st.columns(len(SUGGESTIONS))
        for col, s in zip(cols, SUGGESTIONS):
            if col.button(s, key=f"sug_{s}", width="stretch"):
                ss["pending_q"] = s
                st.rerun()
    for m in ss["msgs"]:
        render_msg(m)

    if q:
        was_new = ss["conv_id"] is None
        user_msg = {"role": "user", "content": q, "citations": None}
        ss["msgs"].append(user_msg)
        render_msg(user_msg)
        with st.spinner("Thinking..."):
            r = call("POST", "/qa/ask", timeout=300, json={
                "question": q, "subject_id": labels.get(subj), "mode": MODES[mode_label],
                "conversation_id": ss["conv_id"]})
        if r.ok:
            a = r.json()
            ss["conv_id"] = a["conversation_id"]
            ss["msgs"].append({"role": "assistant", "content": a["answer"], "citations": a["citations"]})
            if was_new:
                st.rerun()      # refresh the sidebar history list
            render_msg(ss["msgs"][-1])
        else:
            ss["msgs"].pop()
            st.error(err(r))


# ------------------------------------------------------------------ Library
def library():
    ss = st.session_state
    ss.setdefault("up_nonce", 0)
    hero("📚 Library", "Your study material, organised by subject.")
    subjects = call("GET", "/subjects").json()
    labels = {s["name"]: s["id"] for s in subjects}

    with st.expander("➕ New subject", expanded=not subjects):
        with st.form("new_subject", clear_on_submit=True):
            new = st.text_input("Subject name (e.g. Machine Learning)")
            added = st.form_submit_button("Create subject")
        if added and new.strip():
            r = call("POST", "/subjects", json={"name": new.strip()})
            if r.ok:
                st.rerun()
            else:
                st.error(err(r))

    options = ["All documents", *labels]
    if ss.get("lib_subject") not in options:
        ss.pop("lib_subject", None)
    sel = st.selectbox("Subject", options, key="lib_subject")
    sid = labels.get(sel)

    with st.container(border=True):
        st.markdown("##### Upload study material")
        up = st.file_uploader("PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed",
                              key=f"up_{ss['up_nonce']}")
        if sid:
            st.caption(f"Will be added to **{sel}**.")
        else:
            st.warning("Pick a subject above so LearnMate can build a concept map from this file.")
        if up and st.button("Ingest", type="primary", key="ingest"):
            with st.spinner("Reading, splitting and indexing your file..."):
                r = call("POST", "/documents", files={"file": (up.name, up.getvalue())},
                         data={"subject_id": sid} if sid else {}, timeout=600)
            if r.ok and r.json()["status"] == "ready":
                ss["up_nonce"] += 1
                st.rerun()
            elif r.ok:
                st.error(f"{r.json()['filename']} failed: {r.json().get('error')}")
            else:
                st.error(err(r))

    docs = call("GET", "/documents", params={"subject_id": sid} if sid else {}).json()
    st.markdown(f"##### Documents ({len(docs)})")
    if not docs:
        st.caption("Nothing here yet.")
    for d in docs:
        with st.container(border=True):
            a, b = st.columns([8, 1])
            icon = "✅" if d["status"] == "ready" else "❌"
            a.markdown(f"{icon} **{d['filename']}**")
            a.caption(f"{d['page_count']} pages · {d['chunk_count']} passages"
                      + ("" if d["subject_id"] else " · no subject"))
            if b.button("🗑", key=f"deldoc_{d['id']}", help="Delete this document"):
                call("DELETE", f"/documents/{d['id']}")
                st.rerun()

    if sid:
        with st.popover("Delete this subject"):
            st.write("The subject and its concepts are removed; documents are kept (unassigned).")
            if st.button("Yes, delete subject", key="del_subject"):
                call("DELETE", f"/subjects/{sid}")
                st.rerun()


# ------------------------------------------------------------------ Concept map
def dot_graph(cs: list[dict]) -> str:
    esc = lambda s: s.replace('"', "'")   # noqa: E731
    lines = ["digraph G {", "rankdir=LR;", "bgcolor=transparent;",
             'node [shape=box, style="rounded,filled", fontname="Helvetica"];',
             'edge [color="#9aa0b4"];']
    for c in cs:
        fill = "#b7f0c1" if c["verified"] else "#e1dcff"
        lines.append(f'n{c["id"]} [label="{esc(c["name"])}", fillcolor="{fill}", fontcolor="#111111"];')
    for c in cs:
        for p in c["prerequisites"]:
            lines.append(f'n{p["id"]} -> n{c["id"]} [style={"solid" if p["verified"] else "dashed"}];')
    lines.append("}")
    return "\n".join(lines)


def concept_editor(c: dict, others: list[dict]):
    cid = c["id"]

    def act(method, path, **kw):
        r = call(method, path, **kw)
        if r.ok:
            st.rerun()
        else:
            st.error(err(r))

    nm = st.text_input("Name", c["name"], key=f"cn_{cid}")
    ds = st.text_area("Description", c["description"] or "", key=f"cd_{cid}", height=80)
    vf = st.checkbox("✅ I have verified this concept", c["verified"], key=f"cv_{cid}")
    if st.button("Save changes", key=f"cs_{cid}", type="primary"):
        act("PATCH", f"/concepts/{cid}", json={"name": nm, "description": ds, "verified": vf})

    if st.toggle("Show source passages", key=f"ct_{cid}"):
        r = call("GET", f"/concepts/{cid}/chunks")
        for ch in (r.json() if r.ok else []):
            st.caption(f"{ch['filename']} · page {ch['page_number']}")
            st.write(ch["text"])

    st.markdown("**Needs first (prerequisites)**")
    if not c["prerequisites"]:
        st.caption("None yet.")
    for p in c["prerequisites"]:
        a, b, d = st.columns([6, 1, 1])
        a.write(f"{'✅' if p['verified'] else '🤖'} {p['name']}  ·  {p['source']}, conf {p['confidence']}")
        if not p["verified"] and b.button("✔", key=f"pv_{cid}_{p['id']}", help="Mark as correct"):
            act("PATCH", f"/concepts/{cid}/prerequisites/{p['id']}", json={"verified": True})
        if d.button("✖", key=f"pr_{cid}_{p['id']}", help="Remove this link"):
            act("DELETE", f"/concepts/{cid}/prerequisites/{p['id']}")
    have = {p["id"] for p in c["prerequisites"]}
    choices = {o["name"]: o["id"] for o in others if o["id"] != cid and o["id"] not in have}
    if choices:
        a, b = st.columns([4, 1])
        pick = a.selectbox("Add a prerequisite", list(choices), key=f"pa_{cid}", label_visibility="collapsed")
        if b.button("Add", key=f"pb_{cid}"):
            act("POST", f"/concepts/{cid}/prerequisites", json={"prerequisite_id": choices[pick]})

    merge_choices = {o["name"]: o["id"] for o in others if o["id"] != cid}
    a, b, d = st.columns([4, 1, 1])
    if merge_choices:
        into = a.selectbox("Merge into", list(merge_choices), key=f"mg_{cid}", label_visibility="collapsed")
        if b.button("Merge", key=f"mb_{cid}", help="Same idea as the selected concept"):
            act("POST", f"/concepts/{cid}/merge", json={"into_id": merge_choices[into]})
    with d.popover("🗑"):
        if st.button("Delete concept", key=f"cx_{cid}"):
            act("DELETE", f"/concepts/{cid}")


def concepts():
    hero("🧩 Concept map", "The key ideas in your material, and what depends on what.")
    subjects = call("GET", "/subjects").json()
    if not subjects:
        st.info("Create a subject and upload material in the **Library** first.")
        return
    labels = {s["name"]: s["id"] for s in subjects}
    if st.session_state.get("cm_subject") not in labels:
        st.session_state.pop("cm_subject", None)
    sid = labels[st.selectbox("Subject", list(labels), key="cm_subject")]

    docs = [d for d in call("GET", "/documents", params={"subject_id": sid}).json() if d["status"] == "ready"]
    with st.container(border=True):
        st.markdown(f"**{len(docs)}** ready document(s) in this subject.")
        st.caption("Concepts are extracted by your local AI model. Expect a minute or two. "
                   "AI-suggested concepts and links are marked 🤖 until you verify them.")
        pick = st.multiselect("Limit to specific documents (optional)",
                              [f"{d['filename']} (#{d['id']})" for d in docs])
        ids = [int(p.rsplit("#", 1)[1].rstrip(")")) for p in pick] or None
        if st.button("✨ Extract concepts", type="primary", disabled=not docs, key="extract"):
            with st.spinner("Reading your material and finding concepts..."):
                r = call("POST", "/concepts/extract", timeout=3600,
                         json={"subject_id": sid, "document_ids": ids})
            if r.ok:
                x = r.json()
                st.success(f"{x['new_concepts']} new concepts, {x['new_prerequisites']} new prerequisite links "
                           f"({x['chunks_processed']} passages read).")
                if x["batches_failed"]:
                    st.warning(f"{x['batches_failed']} section(s) could not be parsed from the model's reply.")
            else:
                st.error(err(r))

    cs = call("GET", "/concepts", params={"subject_id": sid}).json()
    links = sum(len(c["prerequisites"]) for c in cs)
    a, b, c, d = st.columns(4)
    a.metric("Concepts", len(cs))
    b.metric("Verified", sum(1 for x in cs if x["verified"]))
    c.metric("Prerequisite links", links)
    d.metric("Unverified links", sum(1 for x in cs for p in x["prerequisites"] if not p["verified"]))

    tab_list, tab_graph, tab_add = st.tabs(["📋 Review & correct", "🕸 Graph", "➕ Add manually"])
    with tab_list:
        if not cs:
            st.caption("No concepts yet. Extract them from your documents, or add some manually.")
        term = st.text_input("Search", key="cm_search", placeholder="Filter concepts...")
        shown = [x for x in cs if term.lower() in x["name"].lower()][:40]
        for x in shown:
            title = f"{'✅' if x['verified'] else '🤖'} {x['name']}  ·  {x['chunk_count']} passages"
            with st.expander(title):
                concept_editor(x, cs)
        if len([x for x in cs if term.lower() in x["name"].lower()]) > 40:
            st.caption("Showing the first 40. Use search to narrow down.")
    with tab_graph:
        if links:
            st.graphviz_chart(dot_graph(cs), width="stretch")
            st.caption("Arrow = 'learn this first'. Dashed = AI-suggested, solid = verified. "
                       "Green boxes are verified concepts.")
        else:
            st.caption("No prerequisite links yet.")
    with tab_add:
        with st.form("add_concept", clear_on_submit=True):
            n = st.text_input("Concept name")
            ds = st.text_area("Short description (optional)", height=80)
            ok = st.form_submit_button("Add concept")
        if ok:
            r = call("POST", "/concepts", json={"subject_id": sid, "name": n, "description": ds or None})
            if r.ok:
                st.rerun()
            else:
                st.error(err(r))
