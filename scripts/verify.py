#!/usr/bin/env python3
"""End-to-end evidence script for the local RAG MVP.

Launches its own uvicorn subprocess on a freshly-picked free port, waits for
the index to be ready, drives the API, prints evidence, kills only the process
it started. Exit 0 iff every check passes.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
POLL_INTERVAL = 0.2
HEALTH_TIMEOUT = 120.0

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return ok


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    # Phase 1: launch our own uvicorn on a port nothing else is holding.
    errfile = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    proc: subprocess.Popen | None = None
    port = pick_free_port()
    print(f"[launch] port {port}")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=errfile,
        )
        check("launch: uvicorn subprocess started", True, f"pid {proc.pid}")

        # Phase 2: doctor — the process must answer /health. A fresh index is
        # legitimately empty here: /reindex is the next phase, so requiring
        # chunks>0 now would deadlock the script forever.
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + HEALTH_TIMEOUT
        health = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                resp = httpx.get(f"{base}/health", timeout=5)
                if resp.status_code == 200:
                    health = resp.json()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(POLL_INTERVAL)
        errfile.seek(0)
        err_tail = "".join(errfile.readlines()[-25:])
        if health is None:
            if proc.poll() is not None:
                check("doctor: /health answers", False, f"process exited rc={proc.returncode}\n{err_tail}")
            else:
                check("doctor: /health answers", False, f"timeout after {HEALTH_TIMEOUT}s\n{err_tail}")
            return finish()
        check("doctor: /health answers", True, f"chunks_before={health['chunks']} docs={health['documents']}")

        probe = httpx.get(f"{base}/health", params={"deep": 1}, timeout=120)
        gateway = probe.json().get("gateway", {})
        check("doctor: gateway reachable", bool(gateway.get("ok")), str(gateway.get("detail"))[:200])

        # Phase 3: drive.
        reindex = httpx.post(f"{base}/reindex", timeout=300)
        check("drive: POST /reindex 200", reindex.status_code == 200, reindex.text[:300])
        stats = reindex.json() if reindex.status_code == 200 else {}
        check("drive: index non-empty after reindex", stats.get("chunks", 0) > 0,
              f"documents={stats.get('documents')} chunks={stats.get('chunks')}")

        questions = [
            ("退款多久到账", False, "退款"),
            ("How much does the Pro plan cost", False, None),
            ("火星上的重力加速度是多少", True, None),
            # regression for the measured leak: this scored 0.523, ABOVE the
            # answerable English question, so only the model gate can refuse it.
            ("如何给 PDF 加密", True, None),
        ]
        for q, want_refused, want_path in questions:
            resp = httpx.post(f"{base}/ask", json={"question": q}, timeout=120)
            if not check(f"drive: POST /ask 200 ({q})", resp.status_code == 200, resp.text[:300]):
                continue
            data = resp.json()
            check(f"ask refused={want_refused} ({q})", data["refused"] == want_refused,
                  f"got refused={data['refused']}")
            if want_refused is False and data["refused"] is False:
                paths = " ".join(s["path"] for s in data["sources"])
                if want_path:
                    check(f"source path contains {want_path!r} ({q})", want_path in paths, paths)
                check(f"has sources ({q})", len(data["sources"]) > 0, paths)

            # Phase 4: evidence.
            print(f"\n--- evidence: {q} ---")
            print(f"refused: {data['refused']}  top_score: {data['top_score']}")
            print(f"answer: {data['answer']}")
            for s in data["sources"]:
                print(f"  [{s['n']}] {s['path']}#{s['heading']} score={s['score']}")
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", data["answer"])]
            check(f"cited [n] within range ({q})", all(1 <= n <= len(data["sources"]) for n in cited),
                  f"cited={cited} nsources={len(data['sources'])}")
        return finish()
    finally:
        # Phase 5: cleanup — only the process we started, even if never started.
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"[cleanup] terminated pid {proc.pid}")
        errfile.close()
        Path(errfile.name).unlink(missing_ok=True)


def finish() -> int:
    ok = all(r[1] for r in results)
    passed = sum(r[1] for r in results)
    print(f"\n{'PASS' if ok else 'FAIL'}  summary: {passed}/{len(results)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
