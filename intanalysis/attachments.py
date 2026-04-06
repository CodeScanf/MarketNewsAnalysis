"""Attachment parsing and query-time retrieval helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from intanalysis.embeddings import EmbeddingService, tokenize_text
from intanalysis.models import AttachmentBlock, AttachmentContext, AttachmentEvidence
from text_cleaning import clean_html_text


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PDF_SUFFIXES = {".pdf"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")


def _clean_attachment_text(text: str) -> str:
    cleaned = clean_html_text(text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _chunk_text(text: str, *, page_no: int, block_type: str, prefix: str, max_chars: int = 900) -> list[AttachmentBlock]:
    cleaned = _clean_attachment_text(text)
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    blocks: list[AttachmentBlock] = []
    counter = 1
    for paragraph in paragraphs or [cleaned]:
        if len(paragraph) <= max_chars:
            blocks.append(
                AttachmentBlock(
                    block_id=f"{prefix}_b{counter}",
                    page_no=page_no,
                    block_type=block_type,
                    text=paragraph,
                )
            )
            counter += 1
            continue

        pieces: list[str] = []
        current = ""
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)

        for piece in pieces:
            blocks.append(
                AttachmentBlock(
                    block_id=f"{prefix}_b{counter}",
                    page_no=page_no,
                    block_type=block_type,
                    text=piece,
                )
            )
            counter += 1
    return blocks


class AttachmentParser:
    """Parse uploaded PDF or image files into normalized text blocks."""

    def parse_file(
        self,
        file_path: str | Path,
        *,
        file_name: str | None = None,
        content_type: str | None = None,
    ) -> AttachmentContext:
        path = Path(file_path)
        suffix = path.suffix.lower()
        resolved_name = file_name or path.name

        if suffix in _PDF_SUFFIXES or content_type == "application/pdf":
            return self._parse_pdf(path, resolved_name)
        if suffix in _IMAGE_SUFFIXES or (content_type or "").startswith("image/"):
            return self._parse_image(path, resolved_name)
        raise ValueError(f"Unsupported attachment type: {suffix or content_type or 'unknown'}")

    def _parse_pdf(self, path: Path, file_name: str) -> AttachmentContext:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF parsing backend is not installed. Add 'pypdf' first.") from exc

        reader = PdfReader(str(path))
        blocks: list[AttachmentBlock] = []
        warnings: list[str] = []

        for page_index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            page_blocks = _chunk_text(raw_text, page_no=page_index, block_type="paragraph", prefix=f"p{page_index}")
            if not page_blocks:
                warnings.append(f"Page {page_index} did not yield extractable text.")
            blocks.extend(page_blocks)

        return self._build_context(
            file_name=file_name,
            file_type="pdf",
            page_count=len(reader.pages),
            blocks=blocks,
            warnings=warnings,
        )

    def _parse_image(self, path: Path, file_name: str) -> AttachmentContext:
        warnings: list[str] = []
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin is None:
            warnings.append("Image OCR backend is not configured on the server.")
            return self._build_context(
                file_name=file_name,
                file_type="image",
                page_count=1,
                blocks=[],
                warnings=warnings,
            )

        text = self._run_tesseract(tesseract_bin, path)
        blocks = _chunk_text(text, page_no=1, block_type="ocr_text", prefix="img1")
        if not blocks:
            warnings.append("OCR completed but did not extract readable text.")
        return self._build_context(
            file_name=file_name,
            file_type="image",
            page_count=1,
            blocks=blocks,
            warnings=warnings,
        )

    def _run_tesseract(self, binary: str, path: Path) -> str:
        for languages in ("chi_sim+eng", "eng"):
            try:
                completed = subprocess.run(
                    [binary, str(path), "stdout", "-l", languages],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return completed.stdout
            except subprocess.CalledProcessError:
                continue
        return ""

    @staticmethod
    def _build_context(
        *,
        file_name: str,
        file_type: str,
        page_count: int,
        blocks: Iterable[AttachmentBlock],
        warnings: list[str],
    ) -> AttachmentContext:
        block_list = list(blocks)
        summary = "\n\n".join(block.text for block in block_list[:2]).strip()
        query_text = "\n".join(block.text for block in block_list[:4]).strip()
        return AttachmentContext(
            file_name=file_name,
            file_type=file_type,
            summary=summary[:800],
            query_text=query_text[:2400],
            page_count=page_count,
            blocks=block_list,
            warnings=warnings,
        )


class AttachmentRetriever:
    """Rank parsed attachment blocks for the active query."""

    def __init__(self, embedder: EmbeddingService | None = None):
        self.embedder = embedder or EmbeddingService.get_instance()

    def rank_blocks(
        self,
        query: str,
        attachment_context: AttachmentContext,
        *,
        limit: int = 4,
    ) -> list[tuple[AttachmentBlock, float]]:
        blocks = [block for block in attachment_context.blocks if block.text.strip()]
        if not blocks:
            return []

        query_embedding = self.embedder.embed(query)
        block_embeddings = self.embedder.embed_batch([block.text for block in blocks])
        semantic_scores = np.dot(block_embeddings, query_embedding)

        query_tokens = set(tokenize_text(query))
        scored: list[tuple[AttachmentBlock, float]] = []
        for block, semantic in zip(blocks, semantic_scores):
            block_tokens = set(tokenize_text(block.text[:800]))
            lexical = (
                len(query_tokens & block_tokens) / max(len(query_tokens), 1)
                if query_tokens and block_tokens
                else 0.0
            )
            score = (0.8 * float(semantic)) + (0.2 * lexical)
            scored.append((block, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


def build_attachment_evidence(
    attachment_context: AttachmentContext,
    ranked_blocks: list[tuple[AttachmentBlock, float]],
) -> list[AttachmentEvidence]:
    """Convert ranked blocks into API-friendly evidence objects."""
    evidence: list[AttachmentEvidence] = []
    for block, score in ranked_blocks:
        evidence.append(
            AttachmentEvidence(
                file_name=attachment_context.file_name,
                page_no=block.page_no,
                block_type=block.block_type,
                snippet=block.text[:280],
                score=round(float(score), 4),
                bbox=block.bbox,
                confidence=block.confidence,
            )
        )
    return evidence


def build_attachment_metadata(attachment_context: AttachmentContext) -> dict:
    """Return lightweight metadata safe for chat history storage."""
    return {
        "file_name": attachment_context.file_name,
        "file_type": attachment_context.file_type,
        "page_count": attachment_context.page_count,
        "summary": attachment_context.summary[:240],
        "warning_count": len(attachment_context.warnings),
    }
