# -*- coding: utf-8 -*-

# 格式：
# python eval_rag.py --suite format --retriever full --limit 10 --ignore-source --doc-dir "D:\AI应用开发\test_data_v2" --report "D:\AI应用开发\test_data_v2\自动评测结果\格式对比.json"

# 功能：
# python eval_rag.py --compare --limit 10 --ignore-source --doc-dir "D:\AI应用开发\test_data_v2" --report "D:\AI应用开发\test_data_v2\自动评测结果\消融对比.json"

# python eval_rag.py --compare --ignore-source --doc-dir "D:\AI应用开发\test_data_v2" --report "D:\AI应用开发\test_data_v2\自动评测结果\消融对比.json"



"""
RAG 自动测评脚本 —— 镜像最新版 RAG 代码全功能 + 组合实验套件
"""
# -*- coding: utf-8 -*-
"""
RAG 自动测评脚本 —— 镜像最新版 RAG 代码全功能 + 详细逐题对比
"""
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import re
import sys
import json
import argparse
import collections
import hashlib
from pathlib import Path

import jieba
import docx2txt
import pdfplumber
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

import chromadb
from sentence_transformers import SentenceTransformer

# ==================== 配置区 ====================
DOC_DIR = os.path.dirname(os.path.abspath(__file__))
QUIZ_PATH = os.path.join(DOC_DIR, "RAG测评题库60.json")
PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 50
VECTOR_TOP_K = 5
BM25_TOP_K = 5
RRF_TOP_K = 10
RERANK_TOP_K = 3
RERANK_THRESHOLD = 0.2
RERANK_BATCH_SIZE = 4
DISTANCE_THRESHOLD = 0.6
QUERY_VARIANTS = 3
HISTORY_MESSAGES = 6
HISTORY_ROUNDS = HISTORY_MESSAGES // 2
USE_COSINE = True
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
VARIANT_TEMPERATURE = 0.7
KEYWORD_HIT_RATIO = 0.4
NOT_FOUND_MSG = "抱歉，知识库暂未收录该问题的解决方案。"
ALL_RETRIEVERS = ["vector", "hybrid", "hybrid_rerank", "hybrid_rerank_parent", "full"]
RERANK_MODES = {"hybrid_rerank", "hybrid_rerank_parent", "full"}
SUPPORTED_EXTS = (".docx", ".pdf", ".txt")

# ===========================================================================

def load_env():
    load_dotenv()

def parse_file(path):
    name = os.path.basename(path)
    page_texts = []
    try:
        if name.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        page_texts.append((page_num + 1, text))
        elif name.lower().endswith(".docx"):
            text = docx2txt.process(path)
            if text:
                page_texts.append((1, text))
        elif name.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text:
                page_texts.append((1, text))
    except Exception as e:
        print(f"⚠️ 文件 {name} 解析失败，已跳过：{e}")
    return page_texts

_EMBED_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"加载 Embedding 模型 {EMBED_MODEL}...")
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL)
    return _EMBED_MODEL

def scan_files(doc_dir, files=None):
    if files:
        return [f for f in files if f.lower().endswith(SUPPORTED_EXTS)]
    return sorted(
        str(p) for p in Path(doc_dir).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    )

def group_by_format(paths):
    groups = collections.defaultdict(list)
    for p in paths:
        ext = Path(p).suffix.lower().lstrip(".") or "noext"
        groups[ext].append(p)
    return groups

def build_index(doc_dir, files=None, model=None):
    docs = scan_files(doc_dir, files)
    if not docs:
        raise SystemExit(f"[错误] 没有可解析的 docx/pdf/txt 文档（目录：{doc_dir}）")

    print(f"解析 {len(docs)} 份文档...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)

    all_chunks, all_metas = [], []
    all_parents, all_parent_metas, all_parent_ids = [], [], []
    indexed_names, fmt_stats = [], collections.Counter()
    
    for path in docs:
        name = os.path.basename(path)
        n_child = 0
        for page, text in parse_file(path):
            for p_idx, p_text in enumerate(parent_splitter.split_text(text)):
                pid = hashlib.md5(
                    f"P:{name}:{page}:{p_idx}:{p_text[:50]}".encode("utf-8")).hexdigest()
                all_parents.append(p_text)
                all_parent_metas.append({"source": name, "page": page})
                all_parent_ids.append(pid)
                for c_idx, c_text in enumerate(child_splitter.split_text(p_text)):
                    cid = hashlib.md5(
                        f"C:{pid}:{c_idx}:{c_text[:50]}".encode("utf-8")).hexdigest()
                    all_chunks.append(c_text)
                    all_metas.append({"source": name, "page": page,
                                      "parent_id": pid, "chunk_id": cid})
                    n_child += 1
        fmt_stats[name.rsplit(".", 1)[-1].lower()] += 1
        indexed_names.append(name)
        print(f"  {name}: {n_child} 个子块")

    if not all_chunks:
        raise SystemExit("[错误] 所有文档解析后没有产生任何子块。")

    model = model or get_embed_model()
    client = chromadb.Client()
    
    for _name in ("collection", "collection_parents"):
        try: client.delete_collection(name=_name)
        except Exception: pass
            
    kwargs = {"metadata": {"hnsw:space": "cosine"}} if USE_COSINE else {}
    collection = client.create_collection(name="collection", **kwargs)
    collection.add(
        documents=all_chunks, embeddings=model.encode(all_chunks, batch_size=32).tolist(),
        metadatas=all_metas, ids=[m["chunk_id"] for m in all_metas],
    )
    
    parents_collection = client.create_collection(name="collection_parents")
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    parents_collection.add(
        documents=all_parents, embeddings=[[0.0] * dim for _ in all_parents],
        metadatas=all_parent_metas, ids=all_parent_ids,
    )
    
    all_docs = [Document(page_content=c, metadata=m) for c, m in zip(all_chunks, all_metas)]
    bm25 = BM25Retriever.from_documents(all_docs, preprocess_func=lambda text: list(jieba.cut(text)))
    bm25.k = BM25_TOP_K
    return collection, parents_collection, all_docs, bm25, model, indexed_names

# ==================== 检索管线 ====================
_RERANKER = None

def load_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        print(f"加载重排序模型 {RERANK_MODEL}...")
        _RERANKER = CrossEncoder(RERANK_MODEL)
    return _RERANKER

def get_llm(temperature=LLM_TEMPERATURE):
    # 【修复】加 request_timeout 与 max_retries：原来无超时，OpenAI SDK 默认最长等 10 分钟，
    # 300 次串行调用任一请求挂起就会让评测"卡住没动静"；30 秒超时 + 1 次重试后快速失败降级
    return ChatOpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
                      model=LLM_MODEL, temperature=temperature,
                      request_timeout=30, max_retries=1)

def rewrite_query(question, history):
    if not history: return question
    history_text = "\n".join(f"user: {q}\nassistant: {a[:200]}" for q, a in history)
    prompt = (f"请根据以下对话历史，将用户最新的问题改写成一句独立、完整、便于检索的话。"
              f"只返回改写后的问题，不要添加任何解释、引号或前缀。\n\n对话历史：\n{history_text}\n\n最新问题：{question}\n\n改写后的问题：")
    try:
        rewritten = get_llm().invoke(prompt).content.strip().strip("“”\"'")
        print(f"🔄 查询改写：{question} -> {rewritten}")
        return rewritten if rewritten else question
    except Exception as e:
        return question

def generate_query_variants(query, n=QUERY_VARIANTS):
    prompt = (f"请为以下问题生成 {n} 个不同的查询变体，用于多路检索。\n"
              f"变体应该：\n1. 保持原问题含义不变\n2. 使用不同的表达方式或关键词\n"
              f"3. 每个变体独立成行，不要编号，不要解释\n\n原问题：{query}\n\n生成 {n} 个变体：")
    try:
        text = get_llm(temperature=VARIANT_TEMPERATURE).invoke(prompt).content.strip()
        variants = [v.strip() for v in text.split("\n") if v.strip()]
        return [query] + variants[:n]
    except Exception as e:
        return [query]

def decompose_multi_hop(query):
    """【修复：补齐 app-8 多跳分解的评测能力】判断问题是否多跳，是则拆成 2-3 个依次执行的子查询，否则返回空列表。"""
    prompt = (f"请判断用户的问题是否需要经过两步或两步以上的逻辑推导才能回答（即多跳问题）。"
              f"如果是多跳问题（例如：'小王的上司是谁？'需要先查'小王是哪个部门'，再查'该部门负责人是谁'），"
              f"请将其拆解为 2 个依次执行的子查询；如果不是多跳问题，请返回空列表。"
              f"严格输出 JSON 格式，不要其他文字：{{\"sub_queries\": [\"子查询1\", \"子查询2\"]}}\n\n"
              f"问题：{query}")
    try:
        data = json.loads(get_llm(temperature=VARIANT_TEMPERATURE).invoke(prompt).content.strip())
        subs = data.get("sub_queries", [])
        if isinstance(subs, list) and len(subs) >= 2:
            # 【修复】子查询最多取 3 步，防止 LLM 一次拆出过多子查询导致检索次数与耗时膨胀（与 app-8 一致）
            return [s.strip() for s in subs[:3] if isinstance(s, str) and s.strip()]
    except Exception as e:
        print(f"⚠️ 多跳分解失败，降级为单步检索：{e}")
    return []

def vector_topk(q, collection, model, k=VECTOR_TOP_K):
    vr = collection.query(query_embeddings=model.encode([q]).tolist(), n_results=k)
    return [(Document(page_content=d, metadata=dict(vr["metadatas"][0][i])), vr["distances"][0][i]) for i, d in enumerate(vr["documents"][0])]

def rrf_fusion(vec_groups, bm25_groups, k=60, top_n=RRF_TOP_K):
    """RRF 融合：每一路检索结果内部各自从 0 计 rank（与 qa_flow.py 修复保持一致）。

    旧实现把多路结果拼成长列表后按累积下标当 rank，使第二路及以后的第 1 名
    起点被推后（rank=5、10…），排名被"属于第几路"污染。
    """
    scores, meta_map = collections.defaultdict(float), {}
    for groups in (vec_groups, bm25_groups):
        for group in groups:
            for rank, item in enumerate(group):
                doc = item[0] if isinstance(item, tuple) else item
                # RRF 键从"全文"改为元组 (来源, 页码, 父块ID, 内容前80字)，
                # 避免不同块因内容相同/重叠发生碰撞、metadata 互相覆盖
                key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0),
                       doc.metadata.get("parent_id", ""), doc.page_content[:80])
                scores[key] += 1.0 / (k + rank + 1)
                meta_map[key] = doc
    top = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [meta_map[key] for key in top]

def rerank_scores(query, candidates):
    reranker = load_reranker()
    pairs = [[query, d.page_content] for d in candidates]
    out = []
    for i in range(0, len(pairs), RERANK_BATCH_SIZE):
        out.extend(reranker.predict(pairs[i:i + RERANK_BATCH_SIZE]))
    return out

def backtrace_parents(children, parents_collection):
    parent_ids = list(dict.fromkeys(d.metadata.get("parent_id") for d in children if d.metadata and d.metadata.get("parent_id")))
    if not parent_ids: return []
    try:
        p_res = parents_collection.get(ids=parent_ids)
        id2doc = {pid: (txt, meta) for pid, txt, meta in zip(p_res["ids"], p_res["documents"], p_res["metadatas"])}
        return [Document(page_content=id2doc[pid][0], metadata={**id2doc[pid][1], "parent_id": pid}) for pid in parent_ids if pid in id2doc]
    except Exception: return []

def retrieve(query, collection, parents_collection, bm25, model, mode="hybrid", history=None):
    rerank_mode = mode in RERANK_MODES
    q = rewrite_query(query, history) if mode == "full" else query

    vec_groups, bm25_groups = [], []
    # 【修复：镜像 app-8 多跳迭代检索】full 模式优先尝试多跳分解 + 迭代检索，
    # 把上一步检索结果作为"已知前置信息"拼入下一步；非多跳才走查询变体多路检索
    sub_queries = decompose_multi_hop(q) if mode == "full" else []
    if sub_queries:
        current_context = ""
        for sub_q in sub_queries:
            search_q = f"{sub_q} (已知前置信息: {current_context[:150]})" if current_context else sub_q
            print(f"🔍 多跳迭代检索：{search_q}")
            _vec = vector_topk(search_q, collection, model, VECTOR_TOP_K)
            if _vec:
                vec_groups.append(_vec)
            _bm = [(d, 0.0) for d in bm25.invoke(search_q)]
            if _bm:
                bm25_groups.append(_bm)
            if _vec:
                # 保持原语义：沿用本步检索结果的后两条作为下一步的前置信息（该处疑似 bug，单独记录，本次不改）
                current_context = " ".join(d.page_content[:150] for d, _ in _vec[-2:])
    else:
        queries = generate_query_variants(q) if mode == "full" else [q]
        for qq in queries:
            _vec = vector_topk(qq, collection, model, VECTOR_TOP_K)
            if _vec:
                vec_groups.append(_vec)
            _bm = [(d, 0.0) for d in bm25.invoke(qq)]
            if _bm:
                bm25_groups.append(_bm)
    first_dist = vec_groups[0][0][1] if (vec_groups and vec_groups[0]) else 999.0

    if mode == "vector":
        flat = [d for group in vec_groups for d, _ in group]
        return flat[:RERANK_TOP_K], first_dist, 0.0, q
    fused = rrf_fusion(vec_groups, bm25_groups, top_n=RRF_TOP_K if rerank_mode else RERANK_TOP_K)
    if not rerank_mode: return fused[:RERANK_TOP_K], first_dist, 0.0, q

    try:
        scores = rerank_scores(q, fused)
        top_score = float(max(scores))
        ranked = sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)
        top_children = [d for _, d in ranked[:RERANK_TOP_K]]
    except Exception:
        # 【修复】降级时置 -1.0 而非 1.0：1.0 永远大于拒答阈值 0.2，会让无关文档绕过阈值被送进 LLM；
        # 置 -1.0 必走拒答分支，宁可不答也不错答（与 app-8 修复保持一致）
        top_score, top_children = -1.0, fused[:RERANK_TOP_K]

    # 【新增：双信号兜底】镜像 app-8：Rerank 对问答体/FAQ 语料打分偏低，固定阈值会误杀相关内容。
    # Rerank 低分时用向量余弦距离复核，两路都弱（距离 > 0.6）才拒答，向量信号强则放行。
    if top_score < RERANK_THRESHOLD and fused:
        if first_dist <= DISTANCE_THRESHOLD:
            top_score = RERANK_THRESHOLD + 0.01
            top_children = fused[:RERANK_TOP_K]
            print(f"🛟 双信号兜底放行（评测镜像）：Rerank 低分，向量距离={first_dist:.4f}<={DISTANCE_THRESHOLD}，改用 RRF Top{RERANK_TOP_K}")
        else:
            print(f"🚫 双信号均弱（评测镜像）：Rerank 低分，向量距离={first_dist:.4f}>{DISTANCE_THRESHOLD}，维持拒答")

    final_docs = backtrace_parents(top_children, parents_collection) or top_children if mode in ("hybrid_rerank_parent", "full") else top_children
    return final_docs, first_dist, top_score, q

def build_prompt():
    return PromptTemplate(
        template="""你是一个专业、耐心、严谨的企业 IT 与 HR 专家。请基于以下内部资料回答员工的问题。
资料：
{context}
问题：{question}
要求：
1. 语气要职业、温和。
2. 涉及操作步骤时，请使用 1. 2. 3. 分点列出。
3. 如果资料中找不到答案，请回答“未找到相关信息”，并引导员工转人工。
4. 请在回答末尾标注引用的来源，格式如 [来源: 文件名 第x页]。
5. 如果资料中有对比性、多项数据或列表信息，请用 Markdown 表格呈现。
6. 请完整、逐点覆盖与问题相关的全部要点。资料中没有的内容不要编造。
回答：""", input_variables=["context", "question"])

def answer_llm(query, docs, prompt):
    context = "".join(f"{doc.page_content}\n[来源: {doc.metadata['source']} 第{doc.metadata['page']}页]\n\n" for doc in docs)
    # 【修复】LLM 调用加异常保护：失败时返回"未收录"消息而不是让整个评测挂死，
    # 该题会因缺少答案关键词被判 FAIL，评测继续跑完剩下的题
    try:
        return get_llm().invoke(prompt.format(context=context, question=query)).content
    except Exception as e:
        print(f"⚠️ LLM 回答失败（该题记 FAIL 后继续）：{e}")
        return NOT_FOUND_MSG

# ==================== 判分逻辑 ====================
STOPWORDS = set("的一是在了不和有大这中人上为个所我你他她它们与就都而及把被以但于之其或无")
NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
CLEAN_RE = re.compile(r"[\s，。；：、（）()【】《》“”‘’\"'!？!?\-—]+")

def extract_keywords(text):
    words, nums = [], []
    for tok in jieba.lcut(text):
        t = tok.strip()
        if not t: continue
        if NUM_RE.fullmatch(t) or re.fullmatch(r"[0-9.]+", t): nums.append(t)
        elif len(t) >= 2 and t not in STOPWORDS and not re.fullmatch(r"[\W_]+", t): words.append(t)
    return words, nums

def norm(s): return CLEAN_RE.sub("", s)
def strip_source(reply): return re.sub(r"[\[【]\s*来源[:：][^\]】]*[\]】]", "", reply)
def split_points(answer): return [p.strip() for p in re.split(r"[；;。]|(?<!\d)\.(?!\d)", answer) if len(p.strip()) >= 4] or [answer]

def point_hit(point, uniq, r):
    words, nums = extract_keywords(point)
    if not words and not nums: return norm(point) in r
    num_ok = all(n in r for n in nums) if nums else True
    hit = sum(1 for w in words if w in r)
    kw_ok = (hit / len(words)) >= KEYWORD_HIT_RATIO if words else True
    return num_ok and kw_ok and ((not uniq) or any(w in r for w in uniq))

def match_answer(reply, answer, q_type=None):
    if q_type in ("not_found", "未找到") or "未找到相关信息" in answer:
        return any(k in reply for k in ["未找到", "暂未收录", "未收录"]), "未找到类" if any(k in reply for k in ["未找到", "暂未收录"]) else "答非所问"
    r = norm(strip_source(reply))
    points = split_points(answer)
    all_kw = [set(extract_keywords(p)[0]) for p in points]
    
    # ✅ 修复 bug: 使用 set().union() 安全处理单个要点的情况
    uniqs = []
    for i, kws in enumerate(all_kw):
        others = set().union(*[kw for j, kw in enumerate(all_kw) if j != i])
        uniqs.append(kws - others)
        
    hits = sum(1 for p, u in zip(points, uniqs) if point_hit(p, u, r))
    return (hits / len(points)) >= 0.75, f"命中 {hits}/{len(points)} 点"

# ==================== 详细输出报告 ====================
def print_detailed_results(results, require_source):
    print("\n" + "="*80)
    print(" 🕵️‍♂️ 逐题详细分析报告")
    print("="*80)
    
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n[{r['id']}] {status} | 题型: {r['type']} | 难度: {r['difficulty']}")
        print(f"❓ 问题: {r['question']}")
        print(f"🎯 预期: {r['expected']}")
        
        # 回答分析
        print(f"🤖 回答: {r['reply'][:150].replace(chr(10), ' ')}...")
        if r["passed"]:
            print(f"   -> 判分: {r['reason']}")
        else:
            print(f"   -> 判分: 答案错误 ({r['reason']})")
            if not r["source_ok"] and require_source and r["expected_source"]:
                print(f"   -> 来源: 不匹配！期望 {r['expected_source']}")
                
        # 如果失败或未找到，展示检索到的上下文供分析
        if not r["passed"] and r.get("context"):
            print(f"   📄 检索到的上下文 (前100字):\n      {r['context'][:100].replace(chr(10), ' ')}...")

def run_eval(retriever, quiz, collection, parents, bm25, model, require_source, mode="llm"):
    prompt = build_prompt()
    history_map = collections.defaultdict(list)
    results = []
    # 【修复】每题打印进度：60 题 × LLM 调用需要数分钟，无进度输出会误以为卡死
    for idx, q in enumerate(quiz):
        print(f"[{idx+1}/{len(quiz)}] {q['id']} {q['question'][:22]}", flush=True)
        src_key = tuple(sorted(q.get("expected_source") or [])) or ("none",)
        key = (src_key, q.get("group", "default"))
        history = history_map[key][-HISTORY_ROUNDS:] if retriever == "full" else None
        docs, first_dist, top_score, q_used = retrieve(q["question"], collection, parents, bm25, model, mode=retriever, history=history)

        not_found = (top_score < RERANK_THRESHOLD) if retriever in RERANK_MODES else (first_dist > DISTANCE_THRESHOLD)
        
        if not docs or not_found:
            reply = NOT_FOUND_MSG
            context_text = "无 (检索未命中或分数过低)"
        else:
            context_text = "\n".join([d.page_content for d in docs])
            reply = answer_llm(q_used, docs, prompt) if mode == "llm" else context_text

        answer_ok, reason = match_answer(reply, q["answer"], q.get("type"))
        source_ok = True
        if require_source and q.get("expected_source"):
            source_ok = any(os.path.splitext(s)[0] in reply for s in q["expected_source"])
            
        passed = answer_ok and source_ok
        results.append({
            "id": q["id"], "question": q["question"], "expected": q["answer"], "reply": reply,
            "expected_source": q.get("expected_source") or [], "answer_ok": answer_ok, "reason": reason,
            "source_ok": source_ok, "passed": passed, "capability": q.get("capability", "unknown"),
            "difficulty": q.get("difficulty", "unknown"), "type": q.get("type", "unknown"),
            "context": context_text
        })
        if retriever == "full":
            history_map[key].append((q["question"], reply))
            if len(history_map[key]) > HISTORY_ROUNDS: history_map[key] = history_map[key][-HISTORY_ROUNDS:]

    return results

# ==================== 主流程 ====================
def load_quiz(path, limit=0):
    if not os.path.exists(path): sys.exit(f"[错误] 找不到题库：{path}")
    with open(path, encoding="utf-8") as f: data = json.load(f)
    quiz = data["questions"] if isinstance(data, dict) else data
    return quiz[:limit] if limit > 0 else quiz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm", "retrieval"], default="llm")
    ap.add_argument("--retriever", choices=ALL_RETRIEVERS, default="hybrid")
    ap.add_argument("--compare", action="store_true", help="5 种检索配置逐题对比")
    ap.add_argument("--suite", choices=["format"], default=None)
    ap.add_argument("--doc-dir", default=DOC_DIR)
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--ignore-source", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiz", default=QUIZ_PATH)
    args = ap.parse_args()

    load_env()
    quiz = load_quiz(args.quiz, args.limit)
    require_source = not args.ignore_source

    if args.suite == "format":
        paths = scan_files(args.doc_dir, args.files)
        groups = group_by_format(paths)
        rows = []
        for ext, g in sorted(groups.items()):
            print(f"\n{'='*60}\n📦 实验组：format/{ext} ({len(g)} 份文件)\n{'='*60}")
            try: coll, parents, _, bm25, model, _ = build_index(args.doc_dir, g)
            except SystemExit: continue
            res = run_eval(args.retriever, quiz, coll, parents, bm25, model, require_source, mode=args.mode)
            print_detailed_results(res, require_source)
            rows.append((f"format/{ext}", len(g), coll.count(), sum(r['passed'] for r in res), len(res)))
        
        print("\n" + "="*60 + "\n📊 格式对比汇总\n" + "="*60)
        for name, nf, nc, p, t in rows:
            print(f"{name:<20} | 文件:{nf} | 子块:{nc} | 准确率:{p}/{t} ({p/t*100:.1f}%)")
        return

    # 建库
    collection, parents, _, bm25, model, _ = build_index(args.doc_dir, args.files)

    if not args.compare:
        res = run_eval(args.retriever, quiz, collection, parents, bm25, model, require_source, mode=args.mode)
        print_detailed_results(res, require_source)
        print(f"\n总准确率: {sum(r['passed'] for r in res)}/{len(res)}")
    else:
        print(f"\n########## 消融实验：5 种检索策略逐题交叉对比 ##########")
        all_results = {}
        for cfg in ALL_RETRIEVERS:
            print(f"\n正在测试: {cfg} ...")
            all_results[cfg] = run_eval(cfg, quiz, collection, parents, bm25, model, require_source, mode=args.mode)
            print(f"✅ 完成: {cfg}")
        
        # 打印交叉对比表
        print("\n" + "="*100)
        print(" 📊 逐题消融对比矩阵 (✅=命中, ❌=失败)")
        print("="*100)
        header = f"{'题号':<5} | {'Vector':<8} | {'Hybrid':<8} | {'Rerank':<8} | {'Parent':<8} | {'Full':<8} | 问题简写"
        print(header)
        print("-" * 100)
        
        for i in range(len(quiz)):
            q_id = quiz[i]["id"]
            q_short = quiz[i]["question"][:20] + "..."
            line = f"{q_id:<5} | "
            for cfg in ALL_RETRIEVERS:
                mark = "✅" if all_results[cfg][i]["passed"] else "❌"
                line += f"{mark:<8} | "
            print(line + q_short)
            
        print("\n" + "="*100)
        print(" 📈 汇总准确率")
        for cfg in ALL_RETRIEVERS:
            p = sum(r['passed'] for r in all_results[cfg])
            print(f" {cfg:<20}: {p}/{len(quiz)} ({(p/len(quiz)*100):.1f}%)")

        # 展示未通过的具体原因
        print("\n" + "="*80)
        print(" 🕵️‍♂️ 完整版 (full) 未通过题目详细分析")
        print("="*80)
        fails = [r for r in all_results["full"] if not r["passed"]]
        if not fails:
            print("🎉 太棒了！完整版全部通过！")
        for r in fails:
            print(f"\n[{r['id']}] 问题: {r['question']}")
            print(f"🎯 预期: {r['expected']}")
            print(f"🤖 回答: {r['reply'][:150]}...")
            print(f"⚠️ 原因: {r['reason']}")
            print(f"📄 检索上下文: {r['context'][:150]}...")

        if args.report:
            out = {cfg: {"metrics": {"accuracy": sum(r['passed'] for r in all_results[cfg])/len(quiz)}, "results": all_results[cfg]} for cfg in ALL_RETRIEVERS}
            with open(args.report, "w", encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"\n完整 JSON 报告已导出：{args.report}")

if __name__ == "__main__":
    main()