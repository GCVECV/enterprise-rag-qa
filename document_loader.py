import io
import json
import pdfplumber
import docx2txt
import streamlit as st
from config import HISTORY_FILE
from table_parser import page_body_and_tables


def save_history():
    """把当前对话历史保存到本地 JSON 文件。"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存历史失败：{e}")


def parse_uploaded_file(uploaded_file):
    page_texts = []
    try:
        if uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 表格单独结构化：表头：值 逐行渲染，并从正文里剔除表格区域避免重复
                    body, tbl = page_body_and_tables(page)
                    text = body
                    if tbl:
                        text = (body + "\n【表格】\n" + tbl) if body.strip() else ("【表格】\n" + tbl)
                    if text.strip():
                        page_texts.append({"page": page_num + 1, "text": text})
        elif uploaded_file.name.endswith(".docx"):
            text = docx2txt.process(io.BytesIO(uploaded_file.read()))
            if text:
                page_texts.append({"page": 1, "text": text})
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            if text:
                page_texts.append({"page": 1, "text": text})
    except Exception as e:
        st.warning(f"{uploaded_file.name}解析失败")
    return page_texts
