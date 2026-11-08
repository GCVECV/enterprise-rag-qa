import json
import streamlit as st
import chromadb
import jieba
import config  # 必须在导入 sentence_transformers 之前（config 内部设置 HF_ENDPOINT）
from config import HISTORY_FILE, CHROMA_PATH
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from ingest import run_ingest
from qa_flow import handle_query

st.set_page_config(page_title="企业智能服务台", page_icon="👔", layout="centered")

# ---------------- 界面样式（仅表现层，不影响检索与生成逻辑） ----------------
_CSS = """
<style>
:root{
  --ink:#12232b; --ink-soft:#4a5c66; --line:#e3e8ee;
  --brand:#0f4c5c; --brand-soft:#e7f1f2; --accent:#c98a2e;
  --surface:#ffffff; --canvas:#f6f8f9;
}
html, body, [data-testid="stAppViewContainer"]{
  background:var(--canvas);
  color:var(--ink);
  font-family:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC","Source Han Sans SC",sans-serif;
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{visibility:hidden;}
[data-testid="stSidebar"]{
  background:var(--surface);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] *{font-size:14px;}
.block-container{padding-top:2.4rem; padding-bottom:7rem; max-width:820px;}

.hero{
  padding:26px 30px 24px; border-radius:16px; color:#fff;
  background:linear-gradient(135deg,#0f4c5c 0%,#14606f 58%,#1d7381 100%);
  box-shadow:0 10px 24px rgba(15,76,92,.18);
}
.hero .kicker{font-size:11px; letter-spacing:2.4px; opacity:.72; text-transform:uppercase;}
.hero h1{margin:8px 0 6px; font-size:30px; font-weight:700; letter-spacing:.5px;}
.hero p{margin:0; font-size:14px; line-height:1.7; opacity:.9;}
.hero .rule{width:46px; height:3px; background:var(--accent); border-radius:2px; margin:14px 0 12px; opacity:.95;}

.chips{display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px;}
.chip{
  font-size:12px; padding:6px 11px; border-radius:999px;
  background:var(--surface); border:1px solid var(--line); color:var(--ink-soft);
}
.chip b{color:var(--brand); font-weight:600;}

.panel{
  background:var(--surface); border:1px solid var(--line); border-radius:14px;
  padding:18px 20px; margin:10px 0; box-shadow:0 1px 3px rgba(18,35,43,.04);
}
.panel h3{margin:0 0 6px; font-size:15px; color:var(--brand); font-weight:600;}
.panel p{margin:0; font-size:13px; line-height:1.75; color:var(--ink-soft);}

.kbstat{display:flex; gap:10px; margin:8px 0 4px;}
.kbstat div{
  flex:1; background:var(--brand-soft); border-radius:10px; padding:10px 12px; text-align:center;
}
.kbstat .v{font-size:20px; font-weight:700; color:var(--brand); line-height:1.2;}
.kbstat .l{font-size:11px; color:var(--ink-soft); margin-top:2px;}

.note{font-size:12px; color:var(--ink-soft); line-height:1.7; margin-top:-4px;}

/* 会话气泡 */
[data-testid="stChatMessage"]{
  background:var(--surface); border:1px solid var(--line); border-radius:14px;
  padding:14px 18px; margin:8px 0; box-shadow:0 1px 3px rgba(18,35,43,.04);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
  background:#e8f1f3; border-color:#d3e3e6;
}
[data-testid="stChatMessage"] p{font-size:14px; line-height:1.8;}
[data-testid="stChatMessage"] code{
  background:#f1f4f6; border-radius:6px; padding:1px 5px; font-size:13px;
}

/* 按钮 */
.stButton > button{
  width:100%; border-radius:10px; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); font-size:13px;
  padding:9px 12px; justify-content:center; transition:all .15s ease; box-shadow:none;
}
.stButton > button:hover{border-color:var(--brand); color:var(--brand); background:#fbfcfc;}
.stButton > button [data-testid="stMarkdownContainer"]{width:100%; text-align:center;}
.stButton > button[kind="primary"]{
  background:var(--brand); color:#fff; border-color:var(--brand);
}
.stButton > button[kind="primary"]:hover{background:#0d4351; border-color:#0d4351; color:#fff;}

/* 输入框 */
[data-testid="stChatInput"]{
  border-top:1px solid var(--line); padding-top:12px; background:var(--canvas);
}
[data-testid="stChatInput"] textarea{font-size:14px;}
[data-testid="stFileUploader"]{border-radius:12px;}
[data-testid="stPills"] button{border-radius:999px; font-size:13px; padding:7px 14px;}

#MainMenu, footer{visibility:hidden;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------- 会话状态（与原实现一致） ----------------
if 'messages' not in st.session_state:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        except Exception:
            st.session_state.messages = []
    else:
        st.session_state.messages = []
if 'collection' not in st.session_state:
    st.session_state.collection = None

if 'parents_collection' not in st.session_state:
    st.session_state.parents_collection = None

if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []

if 'model' not in st.session_state:
    with st.spinner("正在加载向量模型 bge-small-zh-v1.5 ..."):
        st.session_state.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None


if st.session_state.collection is None:
    try:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        if any(c.name == "collection" for c in _client.list_collections()):
            _col = _client.get_collection(name="collection")
            if _col.count() > 0:
                st.session_state.collection = _col
                if any(c.name == "collection_parents" for c in _client.list_collections()):
                    st.session_state.parents_collection = _client.get_collection(name="collection_parents")
                _stored = _col.get(include=["documents", "metadatas"])
                st.session_state.all_docs = [
                    Document(page_content=d, metadata=m)
                    for d, m in zip(_stored["documents"], _stored["metadatas"])
                ]
                _bm25 = BM25Retriever.from_documents(
                    st.session_state.all_docs,
                    preprocess_func=lambda text: list(jieba.cut(text)),
                )
                _bm25.k = 5
                st.session_state.bm25_retriever = _bm25
    except Exception as e:
        print(f"⚠️ 加载持久化知识库失败：{e}")

# 知识库规模（供界面展示）
_kb_blocks = len(st.session_state.all_docs)
_kb_sources = len({
    d.metadata.get("source") for d in st.session_state.all_docs if d.metadata.get("source")
})

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.markdown("### 文档管理")
    st.markdown(
        f"""
<div class="kbstat">
  <div><div class="v">{_kb_blocks}</div><div class="l">已入库子块</div></div>
  <div><div class="v">{_kb_sources}</div><div class="l">来源文档</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    if _kb_blocks:
        st.markdown('<div class="note">知识库已就绪，可直接提问。</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="note">知识库为空，请先上传制度文档。</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "上传制度文档",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="支持 PDF / DOCX / TXT。扫描版 PDF 与图片没有 OCR，无法解析。",
    )
    if uploaded_files:
        st.markdown(
            f'<div class="note">已选择 {len(uploaded_files)} 个文件，解析后按 1200 字父块 / 250 字子块入库。</div>',
            unsafe_allow_html=True,
        )
        if st.button("开始解析并存入数据库", type="primary"):
            run_ingest(uploaded_files)

    st.markdown("---")
    with st.expander("维护操作"):
        if st.button("清空知识库"):
            try:
                _c = chromadb.PersistentClient(path=CHROMA_PATH)
                _c.delete_collection(name="collection")
            except Exception:
                pass

            try:
                _c = chromadb.PersistentClient(path=CHROMA_PATH)
                _c.delete_collection(name="collection_parents")
            except Exception:
                pass
            st.session_state.collection = None
            st.session_state.parents_collection = None
            st.session_state.all_docs = []
            st.session_state.bm25_retriever = None
            st.success("知识库已清空，请重新上传文件")
            st.rerun()

        if st.button("清空对话历史"):
            st.session_state.messages = []
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            st.success("已清空对话历史")
            st.rerun()

# ---------------- 主区域 ----------------
st.markdown(
    """
<div class="hero">
  <div class="kicker">Enterprise Knowledge Assistant</div>
  <h1>企业智能服务台</h1>
  <div class="rule"></div>
  <p>面向制度与流程文档的检索增强问答。命中给出出处，资料不足时明确拒答，不靠常识补编。</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="chips">
  <span class="chip">Embedding <b>bge-small-zh-v1.5</b></span>
  <span class="chip">Rerank <b>bge-reranker-base</b></span>
  <span class="chip">生成 <b>deepseek-chat</b></span>
  <span class="chip">检索 <b>向量 + BM25 + RRF</b></span>
  <span class="chip">拒答 <b>阈值 + 向量距离双信号</b></span>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
<div class="panel">
  <h3>先问一个具体问题</h3>
  <p>制度类问题最好带上主体和条件，例如"入职满一年、累计工作 3 年，年假几天"。
  命中条款会标注来源与页码；知识库里没有的内容会直接说明未收录，不会自行编造。</p>
</div>
""",
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🏛️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

st.markdown(
    '<div class="note" style="margin:18px 0 6px;">常见问题：</div>',
    unsafe_allow_html=True,
)

_QUICK_QUESTIONS = ["出差报销标准是多少？", "怎么连公司 VPN？", "年假怎么计算？"]


def _remember_pick():
    st.session_state["_pending_query"] = st.session_state.get("quick_pill")


if "_pending_query" not in st.session_state:
    st.session_state["_pending_query"] = None

st.pills(
    "常见问题",
    _QUICK_QUESTIONS,
    selection_mode="single",
    label_visibility="collapsed",
    key="quick_pill",
    on_change=_remember_pick,
)

quick_query = st.session_state.get("_pending_query")
st.session_state["_pending_query"] = None


query_input = st.chat_input("请输入问题，例如：旷工几天会解除劳动合同？")
query = quick_query or query_input

if query:
    handle_query(query)
