#!/usr/bin/env python3
"""
tool_search_brain.py — OpenClaw 外脑语义检索工具
功能：
  search  语义检索本地知识库（不启动任何服务，仅读取文件）
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output (prevents GBK codec errors on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import numpy as np
except ImportError:
    print("❌ 缺少 numpy，请安装: pip install numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ 缺少 requests，请安装: pip install requests")
    sys.exit(1)


# ─────────────────────────── 工具函数 ───────────────────────────

def decode_key(encoded: str) -> str:
    """解码 Base64 API Key"""
    return base64.b64decode(encoded.encode()).decode()


def load_config(data_dir: str) -> dict:
    """加载配置文件"""
    config_path = Path(data_dir) / "config.json"
    if not config_path.exists():
        print(json.dumps({
            "error": "配置文件不存在，请先运行 notion_sync.py init-config"
        }, ensure_ascii=False))
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────── Embedding（单条查询） ───────────────────────────

def embed_query(text: str, config: dict) -> np.ndarray:
    """将单条查询文本转为向量"""
    provider = config["embedding_provider"]
    api_key = decode_key(config["embedding_api_key"])

    if provider == "azure":
        return _embed_azure(text, api_key, config)
    elif provider == "openai":
        return _embed_openai(text, api_key, config)
    elif provider == "gemini":
        return _embed_gemini(text, api_key, config)
    else:
        raise ValueError(f"不支持的 Embedding 提供商: {provider}")


def _embed_azure(text: str, api_key: str, config: dict) -> np.ndarray:
    endpoint = config.get("azure_endpoint", "").rstrip("/")
    deployment = config.get("azure_deployment", "text-embedding-3-small")
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01"
    resp = requests.post(url, headers={
        "api-key": api_key,
        "Content-Type": "application/json",
    }, json={"input": [text]}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Azure API 错误 [{resp.status_code}]: {resp.text[:200]}")
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def _embed_openai(text: str, api_key: str, config: dict) -> np.ndarray:
    model = config.get("embedding_model", "text-embedding-3-small")
    resp = requests.post("https://api.openai.com/v1/embeddings", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={"input": [text], "model": model}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API 错误 [{resp.status_code}]: {resp.text[:200]}")
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def _embed_gemini(text: str, api_key: str, config: dict) -> np.ndarray:
    model = config.get("embedding_model", "gemini-embedding-001")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"
    resp = requests.post(url, headers={
        "Content-Type": "application/json",
    }, json={"content": {"parts": [{"text": text}]}}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API 错误 [{resp.status_code}]: {resp.text[:200]}")
    return np.array(resp.json()["embedding"]["values"], dtype=np.float32)


# ─────────────────────────── 余弦相似度检索 ───────────────────────────

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算查询向量与矩阵中所有向量的余弦相似度"""
    # 归一化
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


def search_brain(query: str, top_k: int, data_dir: str,
                 output_format: str = "json") -> str:
    """
    语义检索核心函数
    返回 JSON 或 Markdown 格式的结果
    """
    start_time = time.time()

    brain_dir = Path(data_dir) / "my_brain_db"
    vectors_path = brain_dir / "vectors.npy"
    meta_path = brain_dir / "chunks_meta.json"

    # 检查知识库是否存在
    if not vectors_path.exists() or not meta_path.exists():
        return json.dumps({
            "error": "知识库为空，请先运行 notion_sync.py sync 同步数据",
            "results": []
        }, ensure_ascii=False, indent=2)

    # 加载配置和数据
    config = load_config(data_dir)
    vectors = np.load(str(vectors_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 查询向量化
    query_vec = embed_query(query, config)

    # 相似度匹配
    scores = cosine_similarity(query_vec, vectors)
    top_indices = np.argsort(scores)[::-1][:top_k]

    elapsed_ms = int((time.time() - start_time) * 1000)

    # 构造结果
    results = []
    for rank, idx in enumerate(top_indices, 1):
        idx = int(idx)
        score = float(scores[idx])
        item = meta[idx]
        results.append({
            "rank": rank,
            "score": round(score, 4),
            "source_page": item.get("source_page", "未知"),
            "text": item.get("text", ""),
        })

    output = {
        "query": query,
        "results": results,
        "search_time_ms": elapsed_ms,
        "total_chunks": len(meta),
    }

    if output_format == "markdown":
        return format_markdown(output)
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_markdown(output: dict) -> str:
    """将结果格式化为 Markdown"""
    lines = [
        f"## 🔍 检索结果",
        f"**查询**: {output['query']}",
        f"**耗时**: {output['search_time_ms']}ms | "
        f"**知识库文本块**: {output['total_chunks']}",
        "",
    ]
    for r in output["results"]:
        score_bar = "🟢" if r["score"] >= 0.7 else ("🟡" if r["score"] >= 0.5 else "🔴")
        lines.extend([
            f"### {score_bar} #{r['rank']} — {r['source_page']} (相似度 {r['score']})",
            f"> {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}",
            "",
        ])
    return "\n".join(lines)


# ─────────────────────────── 主入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw External Brain - Semantic Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # JSON 格式输出（默认，适合 OpenClaw 解析）
  python tool_search_brain.py search \\
    --query "JUKI 要求的入职印章规格是什么？" \\
    --top-k 3 \\
    --data-dir "./data"

  # Markdown 格式输出（适合直接展示）
  python tool_search_brain.py search \\
    --query "Canon R50 推荐参数" \\
    --top-k 5 \\
    --format markdown \\
    --data-dir "./data"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    p_search = subparsers.add_parser("search", help="语义检索知识库")
    p_search.add_argument("--query", required=True, help="查询文本")
    p_search.add_argument("--top-k", type=int, default=3,
                          help="返回最相关的 K 条结果（默认 3）")
    p_search.add_argument("--format", choices=["json", "markdown"],
                          default="json", help="输出格式（默认 json）")
    p_search.add_argument("--data-dir", required=True, help="数据目录路径")

    args = parser.parse_args()

    if args.command == "search":
        result = search_brain(
            query=args.query,
            top_k=args.top_k,
            data_dir=args.data_dir,
            output_format=args.format,
        )
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
