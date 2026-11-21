from rag.chunker import Chunk, chunk_markdown


def test_longest_chunk_respects_max_chars():
    body = "\n\n".join(
        f"# 第 {i} 节\n\n" + "这是一段足够长的正文内容。" * 40 for i in range(6)
    )
    chunks = chunk_markdown(body, max_chars=200, overlap=20)
    assert chunks
    assert max(len(c.text) for c in chunks) <= 200


def test_nested_heading_path_is_joined():
    body = "# 指南\n\n## 安装\n\n先安装依赖。\n\n### 常见问题\n\n如果安装失败请重试。\n"
    chunks = chunk_markdown(body)
    headings = [c.heading for c in chunks]
    assert "指南 > 安装" in headings
    assert "指南 > 安装 > 常见问题" in headings


def test_empty_text_returns_empty_list():
    assert chunk_markdown("") == []
    assert chunk_markdown("\n\n  \n") == []


def test_doc_without_headings_is_single_chunk_with_empty_heading():
    chunks = chunk_markdown("没有标题的正文。\n第二行。\n")
    assert len(chunks) == 1
    assert chunks[0].heading == ""
    assert chunks[0].ord == 0
    assert isinstance(chunks[0], Chunk)


def test_long_single_paragraph_splits_into_multiple_chunks():
    para = "这是一个没有换行也没有句号分隔的超长段落" * 100
    chunks = chunk_markdown(para, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.heading == "" for c in chunks)
