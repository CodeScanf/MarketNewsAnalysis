"""Tests for attachment parsing and ranking helpers."""

from pathlib import Path

import numpy as np

from intanalysis.attachments import AttachmentParser, AttachmentRetriever
from intanalysis.models import AttachmentBlock, AttachmentContext


class _MockEmbedder:
    def embed(self, text: str):
        lowered = text.lower()
        return np.array(
            [
                1.0 if "回购" in lowered else 0.0,
                1.0 if "银行" in lowered else 0.0,
                float(len(lowered)) / 100.0,
            ],
            dtype=np.float32,
        )

    def embed_batch(self, texts):
        return np.array([self.embed(text) for text in texts], dtype=np.float32)


def test_attachment_parser_extracts_image_text_with_mocked_tesseract(monkeypatch, tmp_path):
    parser = AttachmentParser()
    image_path = tmp_path / "notice.png"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setattr("intanalysis.attachments.shutil.which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr(
        parser,
        "_run_tesseract",
        lambda binary, path: "公司拟以集中竞价方式回购股份，用于员工激励计划。",
    )

    context = parser.parse_file(image_path, file_name="notice.png", content_type="image/png")

    assert context.file_type == "image"
    assert context.page_count == 1
    assert context.blocks
    assert "回购股份" in context.summary


def test_attachment_retriever_prioritizes_relevant_block():
    context = AttachmentContext(
        file_name="notice.pdf",
        file_type="pdf",
        summary="回购公告摘要",
        query_text="回购公告摘要",
        page_count=1,
        blocks=[
            AttachmentBlock(block_id="b1", page_no=1, text="公司拟以集中竞价方式回购股份。"),
            AttachmentBlock(block_id="b2", page_no=1, text="银行贷款余额保持稳定。"),
        ],
    )

    ranked = AttachmentRetriever(embedder=_MockEmbedder()).rank_blocks("请总结回购安排", context, limit=2)

    assert ranked
    assert ranked[0][0].block_id == "b1"
    assert ranked[0][1] >= ranked[1][1]
