"""
Understanding Optics with Python — PDF translation helper.

翻译原则（默认 layout 模式）
- 用原页矢量底图复刻排版与插图：先绘制整页，再仅对「正文」区域做白底遮盖并写入中文。
- 「代码」（t1xtt / t1xbtt / SFTT1000）与「公式 / 数学字体」（CM*、LCIRCLE10）不翻译、不遮盖。
- 机器翻译仅供参考；仅供个人学习使用，请遵守原书版权。

plain 模式：不保留原书排版，导出为连续排版的纯文本 PDF（旧行为）。

依赖：pymupdf, deep-translator；plain 模式另需 fpdf2。

速度说明：多段正文合并后请求 Google；可用 --workers 并行多批请求（默认 4）。
每次 HTTP 翻译耗时写入报告（默认与输出 PDF 同目录、同名 .translate_timing.json）。

示例：
  python translate_optics_pdf.py -i book.pdf -o out.pdf --mode layout
  python translate_optics_pdf.py -i book.pdf -o out.pdf --mode plain --pages 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import fitz
from deep_translator import GoogleTranslator
from fpdf import FPDF

# 合并多条正文一次请求 Google（deep_translator 的 translate_batch 实为逐条，无加速）。
# 使用极少出现的 Unicode 作为分隔符；若返回切分条数不对则自动退回逐条翻译。
JOIN_SEP = "\uFFF9\uFDD0\uFFF9"
DEFAULT_PACK_BUDGET = 4200

CODE_FONTS = frozenset({"t1xtt", "t1xbtt", "SFTT1000"})
REDACT_PAD_PT = 1.0
MIN_BODY_FONTSIZE = 5.0
FONT_SUBSET = "zhsc"


def _span_kind(font: str) -> str:
    if font in CODE_FONTS:
        return "code"
    if font.startswith("CM") or font in {"LCIRCLE10"}:
        return "math"
    return "prose"


def _block_sort_key(block: dict) -> tuple:
    bb = block.get("bbox", (0, 0, 0, 0))
    return (round(bb[1], 2), round(bb[0], 2))


def _span_color_rgb(sp: dict) -> tuple[float, float, float]:
    c = sp.get("color")
    if c is None:
        return (0.0, 0.0, 0.0)
    c = int(c)
    r = ((c >> 16) & 255) / 255.0
    g = ((c >> 8) & 255) / 255.0
    b = (c & 255) / 255.0
    return (r, g, b)


def collect_prose_line_items(page: fitz.Page) -> list[dict]:
    """按阅读顺序收集每一行内连续正文 span（合并 bbox），供 layout 模式擦除与回填。"""
    data = page.get_text("dict")
    blocks = [b for b in data.get("blocks", []) if b.get("type") == 0]
    blocks.sort(key=_block_sort_key)
    out: list[dict] = []

    for bl in blocks:
        for line in bl.get("lines", []):
            spans = line.get("spans", [])
            i = 0
            while i < len(spans):
                sp = spans[i]
                font = sp.get("font") or ""
                if _span_kind(font) != "prose":
                    i += 1
                    continue
                parts: list[str] = [sp.get("text") or ""]
                rect = fitz.Rect(sp["bbox"])
                fsize = float(sp.get("size") or 11)
                color = _span_color_rgb(sp)
                i += 1
                while i < len(spans) and _span_kind(spans[i].get("font") or "") == "prose":
                    parts.append(spans[i].get("text") or "")
                    rect |= fitz.Rect(spans[i]["bbox"])
                    fsize = max(fsize, float(spans[i].get("size") or fsize))
                    c2 = _span_color_rgb(spans[i])
                    if c2 != (0.0, 0.0, 0.0):
                        color = c2
                    i += 1
                t = "".join(parts)
                t = re.sub(r"[ \t]+", " ", t)
                if t.strip():
                    out.append({"rect": rect, "text": t, "size": fsize, "color": color})
    return out


def _inflate_rect(r: fitz.Rect, pad: float, clip: fitz.Rect) -> fitz.Rect:
    rr = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
    return rr & clip


def _insert_textbox_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontname: str,
    orig_size: float,
    color: tuple[float, float, float],
    fontfile: str,
) -> None:
    """在矩形内写入文本，必要时缩小字号直到 PyMuPDF 接受排版（rc >= 0）。"""
    fs = max(MIN_BODY_FONTSIZE, min(orig_size, (rect.y1 - rect.y0) * 0.92))
    while fs >= MIN_BODY_FONTSIZE - 0.01:
        rc = page.insert_textbox(
            rect,
            text,
            fontname=fontname,
            fontfile=fontfile,
            fontsize=fs,
            color=color,
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if rc >= 0:
            return
        fs *= 0.9
    page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontfile=fontfile,
        fontsize=MIN_BODY_FONTSIZE,
        color=color,
        align=fitz.TEXT_ALIGN_LEFT,
    )


def extract_page_segments(page: fitz.Page) -> list[dict]:
    """Reading-order spans classified as prose | code | math."""
    data = page.get_text("dict")
    blocks = [b for b in data.get("blocks", []) if b.get("type") == 0]
    blocks.sort(key=_block_sort_key)

    segments: list[dict] = []

    for bl in blocks:
        for line in bl.get("lines", []):
            for sp in line.get("spans", []):
                font = sp.get("font") or ""
                text = sp.get("text") or ""
                if not text:
                    continue
                kind = _span_kind(font)
                if segments and segments[-1]["kind"] == kind:
                    segments[-1]["text"] += text
                else:
                    segments.append({"kind": kind, "text": text})

    # Merge tiny whitespace-only flips
    merged: list[dict] = []
    for seg in segments:
        if merged and seg["kind"] == merged[-1]["kind"]:
            merged[-1]["text"] += seg["text"]
        else:
            merged.append(seg)

    # Strip and drop empty
    out: list[dict] = []
    for seg in merged:
        t = seg["text"]
        if seg["kind"] == "prose":
            t = re.sub(r"[ \t]+", " ", t)
        seg = {"kind": seg["kind"], "text": t}
        if seg["text"].strip() or seg["kind"] != "prose":
            if seg["text"] or seg["kind"] in ("code", "math"):
                out.append(seg)
    return out


def _normalize_for_translate(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # common PDF ligature artifacts
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    return s.strip()


def _chunk_text(s: str, max_len: int = 4200) -> list[str]:
    s = s.strip()
    if not s:
        return []
    if len(s) <= max_len:
        return [s]
    parts: list[str] = []
    buf: list[str] = []
    n = 0
    for para in re.split(r"(\n{2,})", s):
        if not para:
            continue
        if n + len(para) > max_len and buf:
            parts.append("".join(buf))
            buf = [para]
            n = len(para)
        else:
            buf.append(para)
            n += len(para)
    if buf:
        parts.append("".join(buf))
    return parts


class TranslateCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        self.dirty = False
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, val: str) -> None:
        self.data[key] = val
        self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
        self.dirty = False


def _build_pack_batches(
    need: list[tuple[int, str]],
    pack_budget: int,
) -> list[tuple[list[int], list[str]]]:
    packed: list[tuple[list[int], list[str]]] = []
    pos = 0
    while pos < len(need):
        batch_idx: list[int] = []
        batch_raw: list[str] = []
        acc = 0
        while pos < len(need):
            i, raw = need[pos]
            if JOIN_SEP in raw:
                if batch_raw:
                    break
                batch_idx = [i]
                batch_raw = [raw]
                pos += 1
                break
            add = len(raw) + (len(JOIN_SEP) if batch_raw else 0)
            if batch_raw and acc + add > pack_budget:
                break
            batch_idx.append(i)
            batch_raw.append(raw)
            acc += add
            pos += 1
        packed.append((batch_idx, batch_raw))
    return packed


def _translate_one_request(
    translator: GoogleTranslator,
    text: str,
    sleep_s: float,
    events: list[dict] | None = None,
    *,
    packed_segments: int = 1,
) -> str:
    """单次逻辑请求；长文本按 _chunk_text 切段。events 中记录每次成功 HTTP 的耗时。"""
    chunks = _chunk_text(text)
    out_parts: list[str] = []
    call_idx = 0
    for ch in chunks:
        ch = _normalize_for_translate(ch)
        if not ch:
            continue
        call_idx += 1
        zh: str | None = None
        t0 = time.perf_counter()
        attempts = 0
        for attempt in range(5):
            attempts = attempt + 1
            try:
                zh = translator.translate(ch)
            except Exception:
                zh = None
            if zh:
                break
            if sleep_s > 0:
                time.sleep(sleep_s * attempt)
        dt = time.perf_counter() - t0
        out_parts.append(zh if zh else ch)
        if events is not None:
            events.append(
                {
                    "packed_segments": packed_segments,
                    "chunk_in_request": call_idx,
                    "chars": len(ch),
                    "seconds": round(dt, 4),
                    "attempts": attempts,
                    "ok": bool(zh),
                }
            )
    return "".join(out_parts)


def _translate_pack_job(
    item: tuple[list[int], list[str], float],
) -> tuple[list[int], list[str], list[str], list[dict]]:
    """线程池任务：每个任务新建 GoogleTranslator，避免共享状态。"""
    batch_idx, batch_raw, sleep_s = item
    translator = GoogleTranslator(source="en", target="zh-CN")
    events: list[dict] = []
    n = len(batch_raw)

    if n == 1:
        zh = _translate_one_request(
            translator, batch_raw[0], sleep_s, events, packed_segments=1
        )
        return batch_idx, batch_raw, [zh], events

    blob = JOIN_SEP.join(batch_raw)
    zh_blob = _translate_one_request(
        translator, blob, sleep_s, events, packed_segments=n
    )
    segs = zh_blob.split(JOIN_SEP)
    if len(segs) != n:
        segs = [
            _translate_one_request(
                translator, r, sleep_s, events, packed_segments=1
            )
            for r in batch_raw
        ]
    zh_list: list[str] = []
    for j, rk in enumerate(batch_raw):
        seg = segs[j] if j < len(segs) else ""
        if seg and seg.strip():
            zh_list.append(seg)
        else:
            zh_list.append(rk)
    return batch_idx, batch_raw, zh_list, events


def _write_timing_report(
    path: Path,
    *,
    wall_s: float,
    workers: int,
    pack_budget: int,
    calls: list[dict],
    cache_hits: int,
) -> None:
    http_sum = sum(c.get("seconds", 0) for c in calls)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "wall_clock_seconds": round(wall_s, 4),
        "workers": workers,
        "pack_budget": pack_budget,
        "cache_hits": cache_hits,
        "http_calls": len(calls),
        "http_time_sum_seconds": round(http_sum, 4),
        "note": (
            "workers>1 时 wall_clock 为多线程墙钟时间，通常小于 http_time_sum（请求在时间上重叠）。"
            if workers > 1
            else "单线程下 wall_clock 接近各次 HTTP 耗时之和（另含组包等少量开销）。"
        ),
        "calls": calls,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_batches(
    texts: Iterable[str],
    cache: TranslateCache,
    sleep_s: float = 0.0,
    flush_every: int = 80,
    pack_budget: int = DEFAULT_PACK_BUDGET,
    workers: int = 4,
    timing_report: Path | None = None,
) -> list[str]:
    texts = list(texts)
    out: list[str | None] = [None] * len(texts)
    pending_writes = 0
    cache_hits = 0

    def bump_flush() -> None:
        nonlocal pending_writes
        pending_writes += 1
        if pending_writes >= flush_every:
            cache.flush()
            pending_writes = 0

    need: list[tuple[int, str]] = []
    for i, raw in enumerate(texts):
        hit = cache.get(raw)
        if hit is not None:
            out[i] = hit
            cache_hits += 1
        else:
            need.append((i, raw))

    all_call_events: list[dict] = []
    w_workers = max(1, int(workers))
    wall0 = time.perf_counter()

    if need:
        packed = _build_pack_batches(need, pack_budget)
        jobs = [(bidx, braw, sleep_s) for bidx, braw in packed]

        if w_workers <= 1:
            results = [_translate_pack_job(j) for j in jobs]
        else:
            with ThreadPoolExecutor(max_workers=w_workers) as ex:
                results = list(ex.map(_translate_pack_job, jobs))

        seq = 0
        for batch_idx, batch_raw, zh_list, ev in results:
            for e in ev:
                seq += 1
                e["seq"] = seq
            all_call_events.extend(ev)
            for bi, rk, seg in zip(batch_idx, batch_raw, zh_list, strict=True):
                if not (seg and seg.strip()):
                    seg = rk
                cache.set(rk, seg)
                out[bi] = seg
                bump_flush()

    wall_dt = time.perf_counter() - wall0
    cache.flush()

    if timing_report is not None:
        _write_timing_report(
            timing_report,
            wall_s=wall_dt,
            workers=w_workers,
            pack_budget=pack_budget,
            calls=all_call_events,
            cache_hits=cache_hits,
        )

    return [x if x is not None else "" for x in out]


def find_cjk_font() -> str:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        Path(windir) / "Fonts" / "msyh.ttc",
        Path(windir) / "Fonts" / "msyhbd.ttc",
        Path(windir) / "Fonts" / "simsun.ttc",
        Path(windir) / "Fonts" / "simhei.ttf",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No Chinese font found under Windows Fonts. Install Microsoft YaHei or SimSun."
    )


class ZhPDF(FPDF):
    def __init__(self, cjk_font_path: str):
        super().__init__()
        self.cjk_path = cjk_font_path
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font("zh", "", cjk_font_path)

    def write_segment(self, kind: str, text: str) -> None:
        text = text.replace("\r", "")
        if kind == "prose":
            self.set_font("zh", "", 11)
            self.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif kind == "code":
            # Courier 不支持 Unicode（如 •），与正文共用已嵌入的中文字体
            self.set_font("zh", "", 9)
            self.multi_cell(0, 4.5, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        else:  # math
            self.set_font("zh", "", 10)
            self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)


def run_layout(
    input_pdf: Path,
    output_pdf: Path,
    cache_path: Path,
    start: int,
    end: int | None,
    sleep_s: float,
    pack_budget: int,
    workers: int,
    timing_report: Path | None,
) -> None:
    """复刻插图与矢量排版：底图为原页，仅替换正文为中文。"""
    src = fitz.open(str(input_pdf))
    try:
        n = src.page_count
        end = n if end is None else min(int(end), n)
        start = max(0, int(start))
        if start >= end:
            raise ValueError("start 必须小于 end")

        cache = TranslateCache(cache_path)
        font_path = find_cjk_font()

        pages_items: list[list[dict]] = []
        prose_keys: list[str] = []
        item_refs: list[dict] = []

        for i in range(start, end):
            items = collect_prose_line_items(src.load_page(i))
            pages_items.append(items)
            for it in items:
                if _normalize_for_translate(it["text"]):
                    prose_keys.append(it["text"])
                    item_refs.append(it)
                else:
                    it["zh"] = it["text"]

        if prose_keys:
            translated = translate_batches(
                prose_keys,
                cache,
                sleep_s=sleep_s,
                pack_budget=pack_budget,
                workers=workers,
                timing_report=timing_report,
            )
            for it, zh in zip(item_refs, translated, strict=True):
                it["zh"] = zh

        dst = fitz.open()
        try:
            for idx, pno in enumerate(range(start, end)):
                sp = src.load_page(pno)
                r = sp.rect
                dpo = dst.new_page(width=r.width, height=r.height)
                dpo.show_pdf_page(r, src, pno)

                items = pages_items[idx]
                for it in items:
                    rr = _inflate_rect(it["rect"], REDACT_PAD_PT, dpo.rect)
                    if rr.is_empty or rr.width < 0.4 or rr.height < 0.4:
                        continue
                    dpo.add_redact_annot(rr, fill=(1, 1, 1), text=" ")

                dpo.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                for it in items:
                    zh = it.get("zh", it["text"])
                    inner = fitz.Rect(
                        it["rect"].x0 + 0.15,
                        it["rect"].y0 + 0.15,
                        it["rect"].x1 - 0.15,
                        it["rect"].y1 - 0.15,
                    )
                    if inner.width < 0.5 or inner.height < 0.5:
                        inner = it["rect"]
                    _insert_textbox_fit(
                        dpo,
                        inner,
                        zh,
                        FONT_SUBSET,
                        it["size"],
                        it["color"],
                        font_path,
                    )

            dst.save(str(output_pdf), garbage=4, deflate=True, clean=True)
        finally:
            dst.close()
    finally:
        src.close()


def run(
    input_pdf: Path,
    output_pdf: Path,
    cache_path: Path,
    start: int,
    end: int | None,
    sleep_s: float,
    pack_budget: int,
    workers: int,
    timing_report: Path | None,
) -> None:
    doc = fitz.open(str(input_pdf))
    n = doc.page_count
    end = n if end is None else min(end, n)
    start = max(0, start)
    cache = TranslateCache(cache_path)
    font_path = find_cjk_font()

    all_segments: list[dict] = []

    for i in range(start, end):
        page = doc.load_page(i)
        for seg in extract_page_segments(page):
            all_segments.append({"kind": seg["kind"], "text": seg["text"]})
        all_segments.append(
            {
                "kind": "prose",
                "text": f"\n\n——— 第 {i + 1} 页 ———\n\n",
                "skip_translate": True,
            }
        )

    idx_map: list[int] = []
    texts: list[str] = []
    for si, seg in enumerate(all_segments):
        if seg.get("skip_translate"):
            continue
        if seg["kind"] != "prose":
            continue
        if not _normalize_for_translate(seg["text"]):
            continue
        idx_map.append(si)
        texts.append(seg["text"])

    translated = translate_batches(
        texts,
        cache,
        sleep_s=sleep_s,
        pack_budget=pack_budget,
        workers=workers,
        timing_report=timing_report,
    )
    for si, zh in zip(idx_map, translated):
        all_segments[si]["text"] = zh

    pdf = ZhPDF(font_path)
    pdf.add_page()
    pdf.set_font("zh", "", 11)
    title = (
        "Understanding Optics with Python（译文汇编 · plain）\n"
        "正文为机器翻译；代码与公式按字体规则保留；本页为连续重排，不含原书插图与版面。\n\n"
    )
    pdf.multi_cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

    for seg in all_segments:
        if seg["kind"] == "prose" and not seg["text"].strip():
            continue
        pdf.write_segment(seg["kind"], seg["text"])

    pdf.output(str(output_pdf))
    doc.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=("layout", "plain"),
        default="layout",
        help="layout=保留原页插图与排版并替换正文；plain=纯文本重排 PDF",
    )
    ap.add_argument("--cache", type=Path, default=None, help="Translation JSON cache path")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="Exclusive end page index (0-based)")
    ap.add_argument("--pages", type=int, default=None, help="从 start 起处理页数（与 --end 二选一）")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="仅在翻译失败重试时限流（秒）；默认 0 不按条等待",
    )
    ap.add_argument(
        "--pack-budget",
        type=int,
        default=DEFAULT_PACK_BUDGET,
        help="合并翻译时的最大近似字符数（单批），勿超过约 4500",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行翻译批次数；1=串行。过大可能触发 Google 限流",
    )
    ap.add_argument(
        "--timing-log",
        type=Path,
        default=None,
        help="耗时 JSON；默认与输出 PDF 同目录同名 .translate_timing.json",
    )
    ap.add_argument(
        "--no-timing",
        action="store_true",
        help="不写入耗时报告",
    )
    ap.add_argument(
        "--dry-run-pages",
        type=int,
        default=None,
        help="兼容旧参数，等同 --pages",
    )
    args = ap.parse_args()

    in_pdf = args.input
    out_pdf = args.output
    cache = args.cache or out_pdf.with_suffix(".translate_cache.json")

    doc = fitz.open(str(in_pdf))
    n = doc.page_count
    doc.close()

    start = max(0, args.start)
    if args.end is not None:
        end = min(int(args.end), n)
    elif args.dry_run_pages is not None:
        end = min(start + int(args.dry_run_pages), n)
    elif args.pages is not None:
        end = min(start + int(args.pages), n)
    elif args.mode == "layout":
        end = min(start + 10, n)
    else:
        end = n

    if start >= end:
        raise SystemExit("没有可处理的页面（检查 --start / --end / --pages）")

    if args.no_timing:
        timing_path: Path | None = None
    elif args.timing_log is not None:
        timing_path = args.timing_log
    else:
        timing_path = out_pdf.with_name(out_pdf.stem + ".translate_timing.json")

    if args.mode == "layout":
        run_layout(
            input_pdf=in_pdf,
            output_pdf=out_pdf,
            cache_path=cache,
            start=start,
            end=end,
            sleep_s=args.sleep,
            pack_budget=args.pack_budget,
            workers=args.workers,
            timing_report=timing_path,
        )
    else:
        run(
            input_pdf=in_pdf,
            output_pdf=out_pdf,
            cache_path=cache,
            start=start,
            end=end,
            sleep_s=args.sleep,
            pack_budget=args.pack_budget,
            workers=args.workers,
            timing_report=timing_path,
        )
    print(f"Wrote: {out_pdf}")
    print(f"Cache: {cache}")
    if timing_path is not None:
        print(f"Timing: {timing_path}")


if __name__ == "__main__":
    main()
