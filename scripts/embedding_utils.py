#!/usr/bin/env python3
"""
embedding_utils.py — 共享的 Embedding API 调用模块
被 notion_sync.py 和 tool_search_brain.py 共同使用。
"""

import base64
import json
import sys
import time
from pathlib import Path

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


def decode_key(encoded: str) -> str:
    """解码 Base64 API Key"""
    return base64.b64decode(encoded.encode()).decode()


def encode_key(key: str) -> str:
    """Base64 编码 API Key（混淆存储，非加密）"""
    return base64.b64encode(key.encode()).decode()


def load_config(data_dir: str) -> dict:
    """加载配置文件"""
    config_path = Path(data_dir) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(data_dir: str, config: dict):
    """保存配置文件"""
    config_path = Path(data_dir) / "config.json"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def make_request_with_retry(method: str, url: str, headers: dict,
                            json_data: dict = None, max_retries: int = 3):
    """带重试的 HTTP 请求"""
    for attempt in range(max_retries):
        try:
            if method.lower() == "get":
                return requests.get(url, headers=headers, timeout=30)
            else:
                return requests.post(url, headers=headers, json=json_data, timeout=30)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️ 网络请求异常，{attempt+1}秒后重试... ({type(e).__name__})")
                time.sleep(attempt + 1)
            else:
                print(f"  ❌ 网络请求最终失败: {e}")
                return None
    return None


# ─────────────────────────── 批量 Embedding ───────────────────────────

def get_embeddings(texts: list, config: dict) -> np.ndarray:
    """批量向量化（用于同步）"""
    provider = config["embedding_provider"]
    api_key = decode_key(config["embedding_api_key"])

    if provider == "azure":
        return _embed_azure_batch(texts, api_key, config)
    elif provider == "openai":
        return _embed_openai_batch(texts, api_key, config)
    elif provider == "gemini":
        return _embed_gemini_batch(texts, api_key, config)
    else:
        raise ValueError(f"不支持的 Embedding 提供商: {provider}")


def embed_query(text: str, config: dict) -> np.ndarray:
    """单条查询向量化（用于检索）"""
    provider = config["embedding_provider"]
    api_key = decode_key(config["embedding_api_key"])

    if provider == "azure":
        return _embed_azure_single(text, api_key, config)
    elif provider == "openai":
        return _embed_openai_single(text, api_key, config)
    elif provider == "gemini":
        return _embed_gemini_single(text, api_key, config)
    else:
        raise ValueError(f"不支持的 Embedding 提供商: {provider}")


# ─── Azure ───

def _embed_azure_single(text: str, api_key: str, config: dict) -> np.ndarray:
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


def _embed_azure_batch(texts: list, api_key: str, config: dict) -> np.ndarray:
    endpoint = config.get("azure_endpoint", "").rstrip("/")
    deployment = config.get("azure_deployment", "text-embedding-3-small")
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01"

    all_embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = make_request_with_retry("post", url, headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        }, json_data={"input": batch}, max_retries=3)
        if not resp or resp.status_code != 200:
            err = resp.text[:200] if resp else "no response"
            raise RuntimeError(f"Azure Embedding 错误: {err}")
        data = resp.json()
        for item in sorted(data["data"], key=lambda x: x["index"]):
            all_embeddings.append(item["embedding"])
        if i + batch_size < len(texts):
            time.sleep(0.2)

    return np.array(all_embeddings, dtype=np.float32)


# ─── OpenAI ───

def _embed_openai_single(text: str, api_key: str, config: dict) -> np.ndarray:
    model = config.get("embedding_model", "text-embedding-3-small")
    resp = requests.post("https://api.openai.com/v1/embeddings", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={"input": [text], "model": model}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API 错误 [{resp.status_code}]: {resp.text[:200]}")
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def _embed_openai_batch(texts: list, api_key: str, config: dict) -> np.ndarray:
    model = config.get("embedding_model", "text-embedding-3-small")
    url = "https://api.openai.com/v1/embeddings"

    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = make_request_with_retry("post", url, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }, json_data={"input": batch, "model": model}, max_retries=3)
        if not resp or resp.status_code != 200:
            err = resp.text[:200] if resp else "no response"
            raise RuntimeError(f"OpenAI Embedding 错误: {err}")
        data = resp.json()
        for item in sorted(data["data"], key=lambda x: x["index"]):
            all_embeddings.append(item["embedding"])
        if i + batch_size < len(texts):
            time.sleep(0.2)

    return np.array(all_embeddings, dtype=np.float32)


# ─── Gemini ───

def _embed_gemini_single(text: str, api_key: str, config: dict) -> np.ndarray:
    model = config.get("embedding_model", "gemini-embedding-001")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    resp = requests.post(url, headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }, json={"content": {"parts": [{"text": text}]}}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API 错误 [{resp.status_code}]: {resp.text[:200]}")
    return np.array(resp.json()["embedding"]["values"], dtype=np.float32)


def _embed_gemini_batch(texts: list, api_key: str, config: dict) -> np.ndarray:
    model = config.get("embedding_model", "gemini-embedding-001")
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        url = f"{base_url}/models/{model}:batchEmbedContents"
        requests_body = [{"model": f"models/{model}",
                          "content": {"parts": [{"text": t}]}}
                         for t in batch]
        resp = make_request_with_retry("post", url, headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }, json_data={"requests": requests_body}, max_retries=3)
        if not resp or resp.status_code != 200:
            err = resp.text[:200] if resp else "no response"
            raise RuntimeError(f"Gemini Embedding 错误: {err}")
        data = resp.json()
        for emb in data.get("embeddings", []):
            all_embeddings.append(emb["values"])
        if i + batch_size < len(texts):
            time.sleep(0.2)

    return np.array(all_embeddings, dtype=np.float32)


def test_embedding(config: dict) -> bool:
    """验证 Embedding API 连通性"""
    try:
        result = get_embeddings(["测试连接"], config)
        dim = result.shape[1]
        print(f"  ✅ Embedding API 连接成功，向量维度: {dim}")
        return True
    except Exception as e:
        print(f"  ❌ Embedding API 连接失败: {e}")
        return False
