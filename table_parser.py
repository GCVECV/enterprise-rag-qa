"""PDF 表格结构化解析。

背景：pdfplumber 的 page.extract_text() 会把表格单元格按版面顺序拉平成一行行文字，
表头和数据行的对应关系随之丢失；父子分块再按 250 字切开后，一个子块可能只剩表头、
另一个只剩数据值，检索命中了也答不出"这一列的值分别对应哪个字段"。

做法：把每页的表格单独抽出来，渲染成"表头：值"的逐行文本，作为该页正文的补充段落；
同时用 bbox 过滤把表格区域从正文里剔除，避免同一批文字出现两遍、污染 BM25 词频。
分块参数（父块 1200 / 子块 250）保持不变。
"""


import re

# 中文/数字/百分号之间残留的排版空格（"假期类 型"、"1 5天"）会把词切碎，先合并
_CJK_OR_DIGIT = r"\u4e00-\u9fa5\u3000-\u303f0-9%％"
_INNER_SPACE = re.compile(rf"(?<=[{_CJK_OR_DIGIT}])[ \t]+(?=[{_CJK_OR_DIGIT}])")


def _clean(cell):
    s = (cell or "").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return _INNER_SPACE.sub("", s)


def render_table(table):
    """把单个表格渲染成逐行"表头：值"文本；不足两行（没有数据行）时返回空串。"""
    rows = [[_clean(c) for c in row] for row in (table or [])]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    header = rows[0]
    lines = []
    for row in rows[1:]:
        parts = []
        for idx, cell in enumerate(row):
            if not cell:
                continue
            head = header[idx] if idx < len(header) else ""
            parts.append(f"{head}：{cell}" if head else cell)
        if parts:
            lines.append("；".join(parts))
    return "\n".join(lines)


def find_tables(page):
    """兼容不同 pdfplumber 版本：find_tables() 可能返回 TableFinder，也可能直接返回 list。"""
    res = page.find_tables()
    return res.tables if hasattr(res, "tables") else list(res)


def _inside(obj, bboxes):
    cx = (obj["x0"] + obj["x1"]) / 2.0
    cy = (obj["top"] + obj["bottom"]) / 2.0
    for x0, top, x1, bottom in bboxes:
        if x0 - 1 <= cx <= x1 + 1 and top - 1 <= cy <= bottom + 1:
            return True
    return False


def page_body_and_tables(page):
    """返回 (去掉表格区域后的正文, 结构化表格文本)。任一环节失败时退化为原始正文。"""
    body = page.extract_text() or ""
    try:
        tables = find_tables(page)
    except Exception:
        return body, ""

    rendered = [t for t in (render_table(tb.extract()) for tb in tables) if t]
    bboxes = [tb.bbox for tb in tables]

    if bboxes:
        try:
            body = page.filter(lambda obj: not _inside(obj, bboxes)).extract_text() or ""
        except Exception:
            body = page.extract_text() or ""

    return body, "\n".join(rendered)
