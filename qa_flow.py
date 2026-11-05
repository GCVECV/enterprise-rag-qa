import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from config import RERANK_THRESHOLD, VECTOR_DISTANCE_THRESHOLD
from llm import get_llm_client
from query_analysis import rewrite_query, decompose_multi_hop, generate_query_variants
from document_loader import save_history


def handle_query(query):
    """处理一次提问：改写、检索、重排、拒答判断、流式生成，并保存历史。"""
    if st.session_state.collection is None:
        st.warning("请先上传并解析文件")
        answer = None
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        rewritten_query = rewrite_query(query, st.session_state.messages[:-1])

        with st.chat_message("assistant"):
            with st.spinner("正在检索中..."):
                vector_groups = []
                bm25_groups = []
                bm25_retriever = st.session_state.get('bm25_retriever', None)

                sub_queries = decompose_multi_hop(rewritten_query)

                if sub_queries:
                    st.info(f"🧠 检测到跨文件/多跳问题，Agent 已自动拆解步骤：{' -> '.join(sub_queries)}")
                    current_context = ""
                    for idx, sub_q in enumerate(sub_queries):
                        # 第二步起拼入上一步的检索结果作为线索
                        search_q = sub_q
                        if idx > 0 and current_context:
                            search_q = f"{sub_q} (已知前置信息: {current_context[:150]})"

                        print(f"🔍 多跳迭代检索 第 {idx+1} 步：{search_q}")
                        q_emb = st.session_state.model.encode([search_q]).tolist()
                        v_res = st.session_state.collection.query(
                            query_embeddings=q_emb,
                            n_results=5,
                        )
                        _vec_grp = [
                            Document(page_content=doc, metadata=v_res['metadatas'][0][i])
                            for i, doc in enumerate(v_res['documents'][0])
                        ]
                        if _vec_grp:
                            vector_groups.append(_vec_grp)
                        if bm25_retriever is not None:
                            _bm_grp = list(bm25_retriever.invoke(search_q))
                            if _bm_grp:
                                bm25_groups.append(_bm_grp)

                        if v_res['documents'][0]:
                            current_context = " ".join([d[:150] for d in v_res['documents'][0][:2]])
                else:
                    query_variants = generate_query_variants(rewritten_query, n=3)

                    for q in query_variants:
                        q_emb = st.session_state.model.encode([q]).tolist()
                        v_res = st.session_state.collection.query(
                            query_embeddings=q_emb,
                            n_results=5,
                        )
                        _vec_grp = [
                            Document(page_content=doc, metadata=v_res['metadatas'][0][i])
                            for i, doc in enumerate(v_res['documents'][0])
                        ]
                        if _vec_grp:
                            vector_groups.append(_vec_grp)
                        if bm25_retriever is not None:
                            _bm_grp = list(bm25_retriever.invoke(q))
                            if _bm_grp:
                                bm25_groups.append(_bm_grp)


                def rrf_fusion(doc_groups_a, doc_groups_b, k=60, top_n=10):
                    """RRF 融合：每一路检索结果内部各自从 0 计 rank，累加 1/(k+rank+1)。

                    旧实现把多路结果 extend 成一个长列表后统一 enumerate，
                    导致第一路第 1 名 rank=0、第二路第 1 名 rank=5、第三路 rank=10，
                    排名被"属于第几路"污染。这里按路分组，组内重新计 rank。
                    """
                    scores = {}
                    rrf_meta = {}
                    for groups in (doc_groups_a, doc_groups_b):
                        for group in groups:
                            for rank, doc in enumerate(group):
                                key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0),
                                       doc.metadata.get("parent_id", ""), doc.page_content[:80])
                                scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
                                rrf_meta[key] = doc
                    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
                    return [rrf_meta[key] for key in sorted_keys]

                rrf_docs = rrf_fusion(vector_groups, bm25_groups)

                top_score = 0.0
                top_children = []
                try:
                    if rrf_docs:
                        if "reranker" not in st.session_state:
                            with st.spinner("正在加载Rerank模型，请稍等..."):
                                st.session_state.reranker = CrossEncoder('BAAI/bge-reranker-base')
                        reranker = st.session_state.reranker
                        pairs = [[rewritten_query, doc.page_content] for doc in rrf_docs]

                        reranker_scores = []
                        batch_size = 4
                        for i in range(0, len(pairs), batch_size):
                            batch = pairs[i:i+batch_size]
                            batch_scores = reranker.predict(batch)
                            reranker_scores.extend(batch_scores)

                        print(f"📊 子块 Rerank 分数范围：{min(reranker_scores):.4f} ~ {max(reranker_scores):.4f}，top1={max(reranker_scores):.4f}")

                        ranked = sorted(zip(reranker_scores, rrf_docs), key=lambda x: x[0], reverse=True)
                        top_score = float(ranked[0][0])
                        top_children = [doc for _, doc in ranked[:3]]
                except Exception as e:
                    print(f"⚠️ Rerank 失败，降级使用 RRF 结果：{e}")
                    top_score = -1.0
                    top_children = rrf_docs[:3]

                # Rerank 低分时用向量距离复核，双信号均弱才拒答，避免 FAQ 类文本被误杀
                if top_score < RERANK_THRESHOLD and rrf_docs:
                    _top_score_orig = top_score
                    try:
                        _q_emb2 = st.session_state.model.encode(
                            [rewritten_query], normalize_embeddings=True
                        ).tolist()
                        _v2 = st.session_state.collection.query(
                            query_embeddings=_q_emb2,
                            n_results=1,
                            include=["documents", "metadatas", "distances"],
                        )
                        _dist = _v2["distances"][0][0] if _v2["distances"][0] else 1.0
                        if _dist <= VECTOR_DISTANCE_THRESHOLD:
                            top_score = RERANK_THRESHOLD + 0.01
                            top_children = rrf_docs[:3]
                            print(f"🛟 双信号兜底放行：Rerank={_top_score_orig:.4f} < {RERANK_THRESHOLD}，"
                                  f"向量距离={_dist:.4f} <= {VECTOR_DISTANCE_THRESHOLD}，改用 RRF Top3")
                        else:
                            print(f"🚫 双信号均弱，维持拒答：Rerank={_top_score_orig:.4f}，"
                                  f"向量距离={_dist:.4f} > {VECTOR_DISTANCE_THRESHOLD}")
                    except Exception as e2:
                        print(f"⚠️ 双信号复核失败，维持原拒答判断：{e2}")

                final_docs = []
                if top_children and st.session_state.parents_collection is not None:

                    parent_ids = list(dict.fromkeys([
                        d.metadata.get("parent_id") for d in top_children
                        if d.metadata and d.metadata.get("parent_id")
                    ]))
                    if parent_ids:
                        try:
                            p_res = st.session_state.parents_collection.get(ids=parent_ids)
                            id2doc = {
                                pid: (p_text, p_meta)
                                for pid, p_text, p_meta in zip(
                                    p_res["ids"], p_res["documents"], p_res["metadatas"]
                                )
                            }
                            final_docs = [
                                Document(
                                    page_content=id2doc[pid][0],
                                    metadata={**id2doc[pid][1], "parent_id": pid},
                                )
                                for pid in parent_ids if pid in id2doc
                            ]
                            print(f"🧩 父子分块：Top 3 子块 -> 回溯出 {len(final_docs)} 个父块")
                        except Exception as e:
                            print(f"⚠️ 回溯父块失败，直接使用子块：{e}")
                            final_docs = top_children
                else:
                    final_docs = top_children

                reranker_docs = [doc.page_content for doc in final_docs]
                reranker_metas = [doc.metadata for doc in final_docs]

            if not reranker_docs or top_score < RERANK_THRESHOLD:
                answer = "抱歉，知识库暂未收录该问题的解决方案。"
                st.markdown(answer)
                st.warning("您可以尝试换个问法，或者直接联系人工客服。")
                if st.button("📩 一键提交人工工单"):
                    st.success("已为您生成工单，请留意企业微信通知。")
            else:

                context = ""
                for i, doc in enumerate(reranker_docs):
                    page_info = f"[来源: {reranker_metas[i]['source']} 第{reranker_metas[i]['page']}页]"
                    context += f"{doc}\n{page_info}\n\n"
                prompt_template = PromptTemplate(
                    template="""你是一个专业、耐心、严谨的企业 IT 与 HR 专家。
请基于以下内部资料回答员工的问题。
资料：
{context}
问题：{question}
要求：
1. 语气要职业、温和。
2. 涉及操作步骤时，请使用 1. 2. 3. 分点列出。
3. 如果资料中找不到答案，请回答“未找到相关信息”，并引导员工转人工。
4. 请在回答末尾标注引用的来源，格式如 [来源: 文件名 第x页]。
5. 如果资料中有对比性、多项数据或列表信息，请用 Markdown 表格呈现。
6. 请完整、逐点覆盖与问题相关的全部要点，宁可多列也不要遗漏；资料中没有的内容不要编造。
回答：""",
                    input_variables=["context", "question"],
                )

                final_prompt = prompt_template.format(context=context, question=rewritten_query)

                try:
                    client = get_llm_client()
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        temperature=0.1,
                        messages=[{"role": "user", "content": final_prompt}],
                        stream=True,
                    )
                    answer = st.write_stream(
                        chunk.choices[0].delta.content
                        for chunk in response
                        if chunk.choices[0].delta.content is not None
                    )
                except Exception as e:
                    answer = f"⚠️ 生成回答失败，请稍后重试（{e}）"
                    st.markdown(answer)

    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_history()
