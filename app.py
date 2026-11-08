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
  --brand:#0d4f5c; --brand-deep:#083c46; --brand-soft:#e9f3f4; --brand-line:#cfe2e5;
  --accent:#b8842f;
  --ink:#16262e; --muted:#5d7078; --faint:#93a4ac;
  --line:#e6ebee; --surface:#ffffff; --canvas:#f7f9fa;
}
html, body, [data-testid="stAppViewContainer"]{
  background:var(--canvas); color:var(--ink);
  font-family:"Inter","PingFang SC","Microsoft YaHei UI","Microsoft YaHei",system-ui,sans-serif;
  font-variant-numeric:tabular-nums;
  -webkit-font-smoothing:antialiased;
}
[data-testid="stHeader"]{background:transparent; border-bottom:none;}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:780px; padding:2.1rem 1.5rem 8rem;}

/* 顶部主卡 */
.hero{
  background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:24px 26px 20px;
  box-shadow:inset 0 3px 0 var(--brand), 0 1px 2px rgba(16,38,46,.03), 0 14px 30px -24px rgba(16,38,46,.55);
}
.hero .eyebrow{
  display:flex; align-items:center; gap:9px;
  font-size:10.5px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--brand); font-weight:600;
}
.hero .eyebrow::before{content:""; width:18px; height:2px; background:var(--accent); border-radius:1px;}
.hero h1{margin:13px 0 9px; font-size:30px; line-height:1.25; font-weight:700; letter-spacing:.01em;}
.hero .lede{margin:0; max-width:58ch; font-size:13.5px; line-height:1.85; color:var(--muted);}
.hero-stats{
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px; margin-top:20px; padding-top:18px; border-top:1px solid var(--line);
}
.hero-stat .v{font-size:19px; font-weight:700; color:var(--brand); line-height:1.2;}
.hero-stat .l{font-size:10.5px; color:var(--muted); margin-top:3px;}

/* 技术栈标签 */
.stack{display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin:16px 1px 0;}
.stack .cap{font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--faint); margin-right:2px;}
.tag{font-size:12px; color:var(--muted); background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:5px 11px;}
.tag b{color:var(--ink); font-weight:600;}

/* 知识库状态 */
.status{display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--muted); margin:15px 2px 0;}
.status .dot{width:7px; height:7px; border-radius:50%; background:var(--accent);}
.status.ok .dot{background:#2f8a6b;}

/* 小标题 */
.sec{font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); margin:24px 2px 9px;}

/* 示例问法 */
.examples{display:grid; gap:9px; margin-bottom:2px;}
.ex-card{
  display:flex; gap:14px; align-items:flex-start;
  background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:13px 15px;
}
.ex-card .no{font-size:10.5px; font-weight:700; color:var(--accent); letter-spacing:.06em; padding-top:3px;}
.ex-card .q{font-size:13.5px; color:var(--ink); line-height:1.6;}
.ex-card .meta{font-size:11px; color:var(--muted); margin-top:4px;}

/* 会话气泡 */
[data-testid="stChatMessage"]{
  background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:13px 17px; margin:7px 0; box-shadow:none;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
  background:var(--brand-soft); border-color:var(--brand-line);
  margin-left:12%; width:88%;
}
[data-testid^="stChatMessageAvatar"]{display:none;}
[data-testid="stChatMessage"] p{font-size:13.5px; line-height:1.85; margin:0 0 .55em;}
[data-testid="stChatMessage"] p:last-child{margin-bottom:0;}
[data-testid="stChatMessage"] code{background:#f2f5f7; border:1px solid var(--line); border-radius:5px; padding:1px 5px; font-size:12.5px;}
[data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol{margin:.3em 0 .6em; padding-left:1.2em;}
[data-testid="stChatMessage"] li{font-size:13.5px; line-height:1.8;}

/* 按钮 */
.stButton > button{
  width:100%; border-radius:10px; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); font-size:12.5px; font-weight:500;
  padding:8px 12px; justify-content:center; transition:border-color .15s, color .15s, background .15s; box-shadow:none;
}
.stButton > button:hover{border-color:var(--brand); color:var(--brand); background:#fff;}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:var(--brand); border-color:var(--brand); color:#fff; font-weight:600;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover{
  background:var(--brand-deep); border-color:var(--brand-deep); color:#fff;
}

/* 侧边栏 */
[data-testid="stSidebar"]{background:var(--surface); border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{font-size:12.5px;}
[data-testid="stSidebar"] h3{font-size:13px; font-weight:600; letter-spacing:.01em; margin:2px 0 10px;}
[data-testid="stSidebar"] hr{margin:16px 0; border-color:var(--line);}
.kb-head{display:flex; align-items:center; gap:10px; margin:4px 0 15px;}
.kb-head .mark{
  width:30px; height:30px; border-radius:8px; background:var(--brand); color:#fff;
  display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;
}
.kb-head .t{font-size:13px; font-weight:600; color:var(--ink); line-height:1.25;}
.kb-head .s{font-size:9.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); margin-top:2px;}
.kbstat{
  display:grid; grid-template-columns:1fr 1fr;
  border:1px solid var(--line); border-radius:11px; overflow:hidden; background:var(--surface);
}
.kbstat > div{padding:11px 14px;}
.kbstat > div + div{border-left:1px solid var(--line);}
.kbstat .v{font-size:18px; font-weight:700; color:var(--brand); line-height:1.15;}
.kbstat .l{font-size:10.5px; color:var(--muted); margin-top:3px;}
.note{font-size:12px; color:var(--muted); line-height:1.7;}
[data-testid="stFileUploaderDropzone"]{background:#fbfcfd; border:1px dashed #d5dee3; border-radius:10px; padding:12px;}
[data-testid="stFileUploaderDropzone"] p, [data-testid="stFileUploaderDropzone"] small{font-size:11.5px;}
[data-testid="stExpander"]{border:1px solid var(--line); border-radius:10px; background:var(--surface);}
[data-testid="stExpander"] summary{font-size:12px; padding:8px 12px;}

/* 快捷提问 */
[data-testid="stButtonGroup"]{gap:8px; row-gap:8px;}
[data-testid="stButtonGroup"] button{
  border-radius:999px; border:1px solid var(--line); background:var(--surface);
  color:var(--muted); font-size:12.5px; padding:6px 14px; box-shadow:none;
}
[data-testid="stButtonGroup"] button:hover{border-color:var(--brand); color:var(--brand);}
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[kind="pillsActive"]{
  background:var(--brand-soft); border-color:var(--brand-line); color:var(--brand); font-weight:600;
}

/* 输入区 */
[data-testid="stBottom"]{background:transparent;}
[data-testid="stBottom"] > div{background:linear-gradient(180deg, rgba(247,249,250,0) 0%, var(--canvas) 36%); padding-bottom:.5rem;}
[data-testid="stChatInput"]{
  max-width:780px; margin:0 auto; background:var(--surface);
  border:1px solid var(--line); border-radius:14px;
  box-shadow:0 12px 26px -22px rgba(16,38,46,.8);
}
[data-testid="stChatInput"] > div{border:none; background:transparent;}
[data-testid="stChatInput"] textarea{font-size:13.5px; padding:12px 14px;}
[data-testid="stChatInput"]:focus-within{border-color:var(--brand); box-shadow:0 0 0 3px rgba(13,79,92,.08);}
[data-testid="stChatInputSubmitButton"]{
  background:var(--brand); color:#fff; border-radius:50%;
  width:32px; height:32px; min-width:32px; margin:6px 8px 6px 0;
}
[data-testid="stChatInputSubmitButton"]:hover{background:var(--brand-deep); color:#fff;}
[data-testid="stChatInputSubmitButton"] svg{fill:#fff; stroke:#fff;}

/* 窄屏 */
@media (max-width:640px){
  .block-container{padding:1.4rem 1rem 7rem;}
  .hero{padding:19px 18px 16px;}
  .hero h1{font-size:23px;}
  .hero .lede{font-size:12.5px;}
  .hero-stats{grid-template-columns:1fr 1fr; gap:12px 10px;}
  .hero-stat .v{font-size:17px;}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){margin-left:0; width:100%;}
}
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

@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    """进程级单例：多个浏览器会话共用同一个向量模型，避免每次新会话重复加载。"""
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')


if 'model' not in st.session_state:
    with st.spinner("正在加载向量模型 bge-small-zh-v1.5 ..."):
        st.session_state.model = _load_embedding_model()

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
    st.markdown(
        """
<div class="kb-head">
  <div class="mark">KB</div>
  <div><div class="t">知识库</div><div class="s">Knowledge base</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="kbstat">
  <div><div class="v">{_kb_blocks}</div><div class="l">已入库片段</div></div>
  <div><div class="v">{_kb_sources}</div><div class="l">来源文档</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    if _kb_blocks:
        st.markdown('<div class="status ok"><span class="dot"></span>已就绪，可直接提问</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status"><span class="dot"></span>为空，先上传制度文档</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "上传制度文档",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="支持 PDF / DOCX / TXT。扫描版 PDF 与图片没有 OCR，无法解析。",
    )
    if uploaded_files:
        st.markdown(
            f'<div class="note">已选 {len(uploaded_files)} 个文件，按 1200 字父块 / 250 字子块入库。</div>',
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
  <div class="eyebrow">Enterprise Knowledge Assistant</div>
  <h1>企业制度问答</h1>
  <p class="lede">面向员工手册、考勤休假与信息安全制度的检索增强问答。回答回溯到条款与页码，资料不足时明确拒答，不用常识补编。</p>
  <div class="hero-stats">
    <div class="hero-stat"><div class="v">60/60</div><div class="l">开发集回归</div></div>
    <div class="hero-stat"><div class="v">213/240</div><div class="l">独立集通过</div></div>
    <div class="hero-stat"><div class="v">40/40</div><div class="l">拒答准确</div></div>
    <div class="hero-stat"><div class="v">0.796</div><div class="l">检索 MRR@3</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="stack">
  <span class="cap">实现</span>
  <span class="tag">向量 <b>bge-small-zh-v1.5</b></span>
  <span class="tag">重排 <b>bge-reranker-base</b></span>
  <span class="tag">生成 <b>deepseek-chat</b></span>
  <span class="tag">检索 <b>向量 + BM25 + RRF</b></span>
  <span class="tag">拒答 <b>阈值 + 向量距离双信号</b></span>
</div>
""",
    unsafe_allow_html=True,
)

if _kb_blocks:
    st.markdown(
        f'<div class="status ok"><span class="dot"></span>知识库 {_kb_blocks} 个片段 · {_kb_sources} 份文档，可直接提问</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status"><span class="dot"></span>知识库为空：先在左侧上传三份制度文档，再开始提问</div>',
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown(
        """
<div class="sec">示例问法</div>
<div class="examples">
  <div class="ex-card"><div class="no">01</div><div><div class="q">入职满一年、累计工作 3 年，年假有几天？</div><div class="meta">多跳推理 · 需要跨条款拼条件</div></div></div>
  <div class="ex-card"><div class="no">02</div><div><div class="q">出差住宿的报销上限是多少？</div><div class="meta">表格抽取 · 附录里的结构化数据</div></div></div>
  <div class="ex-card"><div class="no">03</div><div><div class="q">把客户资料发到私人邮箱算违规吗？</div><div class="meta">信息安全条款 · 需要引用原文判断</div></div></div>
</div>
""",
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown('<div class="sec">快捷提问</div>', unsafe_allow_html=True)

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
