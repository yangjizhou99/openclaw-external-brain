#!/usr/bin/env python3
"""
notion_sync.py — Notion 知识库夜间同步脚本
功能：
  init-config  初始化/更新 API 配置
  sync         拉取 Notion 数据库 → 切块 → 向量化 → 存本地
  status       查看同步状态
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
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


# ─────────────────────────── 常量 ───────────────────────────

CHUNK_SIZE = 500       # 每个文本块的最大字符数
CHUNK_OVERLAP = 50     # 块间重叠字符数
NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"

# Embedding 模型配置映射
EMBEDDING_CONFIGS = {
    "azure": {
        "default_model": "text-embedding-3-small",
        "dimensions": 1536,
    },
    "openai": {
        "default_model": "text-embedding-3-small",
        "dimensions": 1536,
        "endpoint": "https://api.openai.com/v1/embeddings",
    },
    "gemini": {
        "default_model": "gemini-embedding-001",
        "dimensions": 3072,
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
    },
}


# ─────────────────────────── 工具函数 ───────────────────────────

def encode_key(key: str) -> str:
    """Base64 编码 API Key（非明文存储）"""
    return base64.b64encode(key.encode()).decode()


def decode_key(encoded: str) -> str:
    """解码 Base64 API Key"""
    return base64.b64decode(encoded.encode()).decode()


def load_config(data_dir: str) -> dict:
    """加载配置文件"""
    config_path = Path(data_dir) / "config.json"
    if not config_path.exists():
        print("❌ 配置文件不存在，请先运行 init-config")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(data_dir: str, config: dict):
    """保存配置文件"""
    config_path = Path(data_dir) / "config.json"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ─────────────────────────── Notion API ───────────────────────────

def notion_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }


def _make_request_with_retry(method: str, url: str, headers: dict, json_data: dict = None, max_retries: int = 3):
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


def fetch_notion_database(token: str, db_id: str) -> list:
    """获取数据库中的所有页面"""
    headers = notion_headers(token)
    url = f"{NOTION_BASE_URL}/databases/{db_id}/query"
    pages = []
    payload = {"page_size": 100}
    has_more = True

    while has_more:
        resp = _make_request_with_retry("post", url, headers, json_data=payload)
        if not resp:
            break
        if resp.status_code != 200:
            print(f"❌ Notion API 错误 [{resp.status_code}]: {resp.text[:200]}")
            sys.exit(1)
        data = resp.json()
        pages.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        if has_more:
            payload["start_cursor"] = data["next_cursor"]

    return pages


def fetch_page_content(token: str, page_id: str) -> str:
    """获取单个页面的全部文本内容（递归读取所有 block）"""
    headers = notion_headers(token)
    url = f"{NOTION_BASE_URL}/blocks/{page_id}/children?page_size=100"
    texts = []
    
    pages_fetched = 0
    while url and pages_fetched < 20:
        pages_fetched += 1
        resp = _make_request_with_retry("get", url, headers)
        if not resp:
            break
        if resp.status_code != 200:
            print(f"  ⚠️ 读取页面 {page_id} 失败: {resp.status_code}")
            break
        data = resp.json()
        for block in data.get("results", []):
            text = extract_block_text(block)
            if text:
                texts.append(text)
        if data.get("has_more") and data.get("next_cursor"):
            url = f"{NOTION_BASE_URL}/blocks/{page_id}/children?page_size=100&start_cursor={data['next_cursor']}"
        else:
            url = None

    if pages_fetched >= 20:
        print(f"  ⚠️ 达到最大页数限制，停止拉取页面 {page_id}")

    return "\n".join(texts)


def extract_block_text(block: dict) -> str:
    """从 Notion block 中提取纯文本"""
    block_type = block.get("type", "")
    block_data = block.get(block_type, {})

    # 大多数文本类型的 block 都有 rich_text 字段
    rich_text = block_data.get("rich_text", [])
    if rich_text:
        return "".join(rt.get("plain_text", "") for rt in rich_text)

    # 特殊处理 child_page, child_database 等
    if block_type == "child_page":
        return block_data.get("title", "")
    if block_type == "child_database":
        return block_data.get("title", "")

    return ""


def get_page_title(page: dict) -> str:
    """从 Notion 页面对象中提取标题"""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_items = prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in title_items)
    return "Untitled"


def extract_page_properties(page: dict) -> str:
    """将 Notion 页面的数据库属性序列化为可读文本"""
    props = page.get("properties", {})
    lines = []

    for prop_name, prop_data in props.items():
        prop_type = prop_data.get("type", "")
        value = _extract_property_value(prop_type, prop_data)
        if value:  # 跳过空值
            lines.append(f"{prop_name}: {value}")

    return "\n".join(lines)


def _extract_property_value(prop_type: str, prop_data: dict) -> str:
    """根据属性类型提取值"""
    try:
        if prop_type == "title":
            items = prop_data.get("title", [])
            return "".join(t.get("plain_text", "") for t in items)

        elif prop_type == "rich_text":
            items = prop_data.get("rich_text", [])
            return "".join(t.get("plain_text", "") for t in items)

        elif prop_type == "number":
            val = prop_data.get("number")
            return str(val) if val is not None else ""

        elif prop_type == "select":
            sel = prop_data.get("select")
            return sel.get("name", "") if sel else ""

        elif prop_type == "multi_select":
            items = prop_data.get("multi_select", [])
            return ", ".join(item.get("name", "") for item in items)

        elif prop_type == "status":
            st = prop_data.get("status")
            return st.get("name", "") if st else ""

        elif prop_type == "date":
            date_obj = prop_data.get("date")
            if not date_obj:
                return ""
            start = date_obj.get("start", "")
            end = date_obj.get("end", "")
            return f"{start} ~ {end}" if end else start

        elif prop_type == "checkbox":
            return "Yes" if prop_data.get("checkbox") else "No"

        elif prop_type == "url":
            return prop_data.get("url", "") or ""

        elif prop_type == "email":
            return prop_data.get("email", "") or ""

        elif prop_type == "phone_number":
            return prop_data.get("phone_number", "") or ""

        elif prop_type == "people":
            people = prop_data.get("people", [])
            return ", ".join(p.get("name", "") for p in people)

        elif prop_type == "relation":
            relations = prop_data.get("relation", [])
            return f"({len(relations)} linked items)" if relations else ""

        elif prop_type == "formula":
            formula = prop_data.get("formula", {})
            f_type = formula.get("type", "")
            return str(formula.get(f_type, "")) if f_type else ""

        elif prop_type == "rollup":
            rollup = prop_data.get("rollup", {})
            r_type = rollup.get("type", "")
            return str(rollup.get(r_type, "")) if r_type else ""

        elif prop_type == "created_time":
            return prop_data.get("created_time", "")

        elif prop_type == "last_edited_time":
            return prop_data.get("last_edited_time", "")

        elif prop_type == "created_by":
            user = prop_data.get("created_by", {})
            return user.get("name", "")

        elif prop_type == "last_edited_by":
            user = prop_data.get("last_edited_by", {})
            return user.get("name", "")

        elif prop_type == "files":
            files = prop_data.get("files", [])
            return ", ".join(f.get("name", "") for f in files) if files else ""

    except (KeyError, TypeError, AttributeError):
        pass

    return ""


# ─────────────────────────── 文本切块 ───────────────────────────

def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """将长文本按字符数切块，带重叠"""
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # 尝试在句号、换行等位置断开
        if end < text_len:
            for sep in ["\n\n", "\n", "。", ".", "！", "？", "；"]:
                last_sep = text.rfind(sep, start, end)
                # 只有当找到的分隔符使得块的长度大于重叠区域时（即 start 能切实向前推进），才在分隔符处断开
                if last_sep > start and (last_sep + len(sep) - overlap > start):
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            # 强制推进，防止死循环
            next_start = start + 1
        
        start = max(next_start, 0)
        
        if start >= text_len:
            break

    return chunks


# ─────────────────────────── Embedding API ───────────────────────────

def get_embeddings(texts: list, config: dict) -> np.ndarray:
    """调用用户选择的 Embedding API 进行批量向量化"""
    provider = config["embedding_provider"]
    api_key = decode_key(config["embedding_api_key"])

    if provider == "azure":
        return _embed_azure(texts, api_key, config)
    elif provider == "openai":
        return _embed_openai(texts, api_key, config)
    elif provider == "gemini":
        return _embed_gemini(texts, api_key, config)
    else:
        print(f"❌ 不支持的 Embedding 提供商: {provider}")
        sys.exit(1)


def _embed_azure(texts: list, api_key: str, config: dict) -> np.ndarray:
    """Azure OpenAI Embedding"""
    endpoint = config.get("azure_endpoint", "").rstrip("/")
    deployment = config.get("azure_deployment", "text-embedding-3-small")
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01"

    all_embeddings = []
    # Azure 批量限制通常为 16 条
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = _make_request_with_retry("post", url, headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        }, json_data={"input": batch}, max_retries=3)
        if not resp:
            sys.exit(1)
        if resp.status_code != 200:
            print(f"❌ Azure Embedding 错误 [{resp.status_code}]: {resp.text[:200]}")
            sys.exit(1)
        data = resp.json()
        for item in sorted(data["data"], key=lambda x: x["index"]):
            all_embeddings.append(item["embedding"])
        # 节约内存：每批之间短暂暂停
        if i + batch_size < len(texts):
            time.sleep(0.2)

    return np.array(all_embeddings, dtype=np.float32)


def _embed_openai(texts: list, api_key: str, config: dict) -> np.ndarray:
    """OpenAI Embedding"""
    model = config.get("embedding_model", "text-embedding-3-small")
    url = "https://api.openai.com/v1/embeddings"

    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = _make_request_with_retry("post", url, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }, json_data={"input": batch, "model": model}, max_retries=3)
        if not resp:
            sys.exit(1)
        if resp.status_code != 200:
            print(f"❌ OpenAI Embedding 错误 [{resp.status_code}]: {resp.text[:200]}")
            sys.exit(1)
        data = resp.json()
        for item in sorted(data["data"], key=lambda x: x["index"]):
            all_embeddings.append(item["embedding"])
        if i + batch_size < len(texts):
            time.sleep(0.2)

    return np.array(all_embeddings, dtype=np.float32)


def _embed_gemini(texts: list, api_key: str, config: dict) -> np.ndarray:
    """Google Gemini Embedding"""
    model = config.get("embedding_model", "gemini-embedding-001")
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    all_embeddings = []
    # Gemini batchEmbedContents 支持批量
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        url = f"{base_url}/models/{model}:batchEmbedContents?key={api_key}"
        requests_body = [{"model": f"models/{model}",
                          "content": {"parts": [{"text": t}]}}
                         for t in batch]
        resp = _make_request_with_retry("post", url, headers={
            "Content-Type": "application/json",
        }, json_data={"requests": requests_body}, max_retries=3)
        if not resp:
            sys.exit(1)
        if resp.status_code != 200:
            print(f"❌ Gemini Embedding 错误 [{resp.status_code}]: {resp.text[:200]}")
            sys.exit(1)
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


# ─────────────────────────── 命令: init-config ───────────────────────────

def cmd_init_config(args):
    """初始化或更新配置"""
    data_dir = args.data_dir
    config_path = Path(data_dir) / "config.json"

    # 如果是更新模式，先加载现有配置
    if args.update_only and config_path.exists():
        config = load_config(data_dir)
    elif args.append_db and config_path.exists():
        config = load_config(data_dir)
    else:
        config = {}

    # 追加数据库模式
    if args.append_db:
        if args.notion_db_id:
            db_ids = config.get("notion_db_ids", [])
            if args.notion_db_id not in db_ids:
                db_ids.append(args.notion_db_id)
                config["notion_db_ids"] = db_ids
                save_config(data_dir, config)
                print(f"✅ 已添加 Notion 数据库: {args.notion_db_id}")
                print(f"   当前数据库列表: {db_ids}")
            else:
                print(f"ℹ️ 数据库 {args.notion_db_id} 已存在")
        else:
            print("❌ --append-db 需要同时提供 --notion-db-id")
        return

    # 正常初始化 / 更新
    if args.notion_token:
        config["notion_token"] = encode_key(args.notion_token)
    if args.notion_db_id:
        existing_ids = config.get("notion_db_ids", [])
        if args.notion_db_id not in existing_ids:
            existing_ids.append(args.notion_db_id)
        config["notion_db_ids"] = existing_ids
    if args.embedding_provider:
        provider = args.embedding_provider.lower()
        if provider not in EMBEDDING_CONFIGS:
            print(f"❌ 不支持的提供商: {provider}。支持: {list(EMBEDDING_CONFIGS.keys())}")
            sys.exit(1)
        config["embedding_provider"] = provider
        config["embedding_model"] = EMBEDDING_CONFIGS[provider]["default_model"]
        config["embedding_dimensions"] = EMBEDDING_CONFIGS[provider]["dimensions"]
    if args.embedding_api_key:
        config["embedding_api_key"] = encode_key(args.embedding_api_key)
    if args.azure_endpoint:
        config["azure_endpoint"] = args.azure_endpoint
    if args.azure_deployment:
        config["azure_deployment"] = args.azure_deployment

    # 验证必要字段
    required = ["notion_token", "notion_db_ids", "embedding_provider", "embedding_api_key"]
    missing = [r for r in required if r not in config or not config[r]]
    if missing:
        print(f"❌ 缺少必要配置项: {missing}")
        sys.exit(1)

    # Azure 额外验证
    if config["embedding_provider"] == "azure":
        if not config.get("azure_endpoint"):
            print("❌ Azure 提供商需要 --azure-endpoint")
            sys.exit(1)

    save_config(data_dir, config)
    print("✅ 配置已保存到 data/config.json")

    # 验证连通性
    print("\n🔗 验证 Notion API...")
    token = decode_key(config["notion_token"])
    for db_id in config["notion_db_ids"]:
        try:
            resp = requests.get(
                f"{NOTION_BASE_URL}/databases/{db_id}",
                headers=notion_headers(token),
                timeout=10
            )
            if resp.status_code == 200:
                title = resp.json().get("title", [{}])
                db_name = title[0].get("plain_text", "无标题") if title else "无标题"
                print(f"  ✅ 数据库连接成功: {db_name}")
            else:
                print(f"  ❌ 数据库 {db_id} 连接失败 [{resp.status_code}]")
        except Exception as e:
            print(f"  ❌ 数据库 {db_id} 连接异常: {e}")

    print("\n🔗 验证 Embedding API...")
    test_embedding(config)


# ─────────────────────────── 增量同步辅助 ───────────────────────────

def load_existing_sync_data(brain_dir: Path):
    """加载已有的向量和元数据，用于增量同步"""
    vectors_path = brain_dir / "vectors.npy"
    meta_path = brain_dir / "chunks_meta.json"

    if not vectors_path.exists() or not meta_path.exists():
        return None, None

    vectors = np.load(str(vectors_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return vectors, meta


def build_page_edit_index(meta: list) -> dict:
    """从已有元数据构建 {page_id: last_edited_time} 索引"""
    index = {}
    for item in meta:
        pid = item.get("page_id", "")
        edit_time = item.get("last_edited_time", "")
        if pid and edit_time:
            index[pid] = edit_time
    return index


def get_chunks_for_page(page_id: str, meta: list, vectors: np.ndarray):
    """获取某个页面的所有 chunk 元数据和对应向量行"""
    indices = [i for i, m in enumerate(meta) if m.get("page_id") == page_id]
    if not indices:
        return [], None
    page_meta = [meta[i] for i in indices]
    page_vectors = vectors[indices]
    return page_meta, page_vectors


# ─────────────────────────── 命令: sync ───────────────────────────

def cmd_sync(args):
    """执行同步：Notion → 切块 → 向量化 → 本地存储（支持增量）"""
    data_dir = args.data_dir
    config = load_config(data_dir)
    token = decode_key(config["notion_token"])
    db_ids = config.get("notion_db_ids", [])
    full_sync = getattr(args, 'full', False)

    if not db_ids:
        print("Error: no Notion database IDs configured")
        sys.exit(1)

    brain_dir = Path(data_dir) / "my_brain_db"
    brain_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = brain_dir / "vectors.npy"
    meta_path = brain_dir / "chunks_meta.json"

    # ── 加载已有数据（增量同步用）──
    old_vectors, old_meta = None, None
    page_edit_index = {}  # {page_id: last_edited_time}

    if not full_sync:
        old_vectors, old_meta = load_existing_sync_data(brain_dir)
        if old_meta is not None:
            page_edit_index = build_page_edit_index(old_meta)
            print(f"[incremental] loaded {len(page_edit_index)} cached pages")
        else:
            print("[incremental] no cache found, running full sync")

    if full_sync:
        print("[full sync] forcing full re-sync of all pages")

    # ── 备份 ──
    if vectors_path.exists():
        backup_path = brain_dir / "vectors_backup.npy"
        import shutil
        shutil.copy2(str(vectors_path), str(backup_path))
        print("backed up previous vectors")

    # ── 分类页面：new / modified / unchanged ──
    final_chunks = []       # 最终的文本列表
    final_meta = []         # 最终的元数据列表
    final_vectors_parts = []  # 最终的向量列表（待合并）

    new_chunks_to_embed = []   # 需要新向量化的文本
    new_meta_to_embed = []     # 对应的元数据

    stats = {"unchanged": 0, "modified": 0, "new": 0, "deleted": 0}
    seen_page_ids = set()

    print(f"syncing {len(db_ids)} database(s)...\n")

    for db_id in db_ids:
        print(f"  database: {db_id}")
        pages = fetch_notion_database(token, db_id)
        print(f"    {len(pages)} pages found")

        for page in pages:
            page_id = page["id"]
            page_title = get_page_title(page)
            page_edited = page.get("last_edited_time", "")
            seen_page_ids.add(page_id)

            # ── 增量判断 ──
            cached_edit_time = page_edit_index.get(page_id)

            if (not full_sync
                    and cached_edit_time
                    and cached_edit_time == page_edited
                    and old_meta is not None
                    and old_vectors is not None):
                # 未修改 → 直接复用旧向量
                page_meta, page_vecs = get_chunks_for_page(
                    page_id, old_meta, old_vectors)
                if page_meta and page_vecs is not None:
                    final_meta.extend(page_meta)
                    final_vectors_parts.append(page_vecs)
                    stats["unchanged"] += 1
                    print(f"    [skip] {page_title} (unchanged)")
                    continue

            # ── 需要重新处理（新/修改）──
            if cached_edit_time:
                stats["modified"] += 1
                label = "update"
            else:
                stats["new"] += 1
                label = "new"
            print(f"    [{label}] {page_title}")

            # 提取属性文本
            props_text = extract_page_properties(page)
            # 提取正文内容
            body_text = fetch_page_content(token, page_id)

            # 合并：属性 + 正文
            if props_text and body_text.strip():
                content = f"{props_text}\n---\n{body_text}"
            elif props_text:
                content = props_text
            elif body_text.strip():
                content = body_text
            else:
                print(f"      [skip] empty content")
                continue

            chunks = split_text(content)
            for i, chunk in enumerate(chunks):
                new_chunks_to_embed.append(chunk)
                new_meta_to_embed.append({
                    "source_db": db_id,
                    "source_page": page_title,
                    "page_id": page_id,
                    "chunk_index": i,
                    "char_count": len(chunk),
                    "last_edited_time": page_edited,
                })

            print(f"      {len(chunks)} chunks")

        time.sleep(0.5)

    # ── 统计被删除的页面（在 Notion 中已不存在）──
    if old_meta is not None:
        old_page_ids = set(m.get("page_id", "") for m in old_meta)
        deleted_ids = old_page_ids - seen_page_ids
        stats["deleted"] = len(deleted_ids)

    # ── 向量化新/修改的 chunks ──
    if new_chunks_to_embed:
        print(f"\nembedding {len(new_chunks_to_embed)} new chunks...")
        new_vectors = get_embeddings(new_chunks_to_embed, config)
        print(f"done, shape: {new_vectors.shape}")

        # 将文本写入 meta
        for meta_item, chunk_text in zip(new_meta_to_embed, new_chunks_to_embed):
            meta_item["text"] = chunk_text

        final_meta.extend(new_meta_to_embed)
        final_vectors_parts.append(new_vectors)
    else:
        print("\nno new content to embed")

    if not final_meta:
        print("\nwarning: no content found in any database")
        return

    # ── 合并所有向量 ──
    if final_vectors_parts:
        all_vectors = np.concatenate(final_vectors_parts, axis=0)
    else:
        print("\nwarning: no vectors to save")
        return

    # ── 保存 ──
    np.save(str(vectors_path), all_vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, ensure_ascii=False, indent=2)

    # ── 记录同步状态 ──
    total_pages = len(set(m.get("page_id", "") for m in final_meta))
    status = {
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "total_pages": total_pages,
        "total_chunks": len(final_meta),
        "vector_dimensions": int(all_vectors.shape[1]),
        "embedding_provider": config["embedding_provider"],
        "embedding_model": config.get("embedding_model", "unknown"),
        "storage_bytes": vectors_path.stat().st_size + meta_path.stat().st_size,
        "last_sync_stats": stats,
    }
    status_path = brain_dir / "sync_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    # ── 清理备份 ──
    backup_path = brain_dir / "vectors_backup.npy"
    if backup_path.exists():
        backup_path.unlink()

    size_mb = status["storage_bytes"] / (1024 * 1024)
    print(f"\nsync complete!")
    print(f"   pages: {total_pages}")
    print(f"   chunks: {len(final_meta)}")
    print(f"   dimensions: {status['vector_dimensions']}")
    print(f"   storage: {size_mb:.1f} MB")
    print(f"   stats: new={stats['new']}, modified={stats['modified']}, "
          f"unchanged={stats['unchanged']}, deleted={stats['deleted']}")


# ─────────────────────────── 命令: status ───────────────────────────

def cmd_status(args):
    """查看同步状态"""
    data_dir = args.data_dir
    brain_dir = Path(data_dir) / "my_brain_db"
    status_path = brain_dir / "sync_status.json"

    if not status_path.exists():
        print("🧠 外脑状态：尚未同步过")
        print("   请运行: python notion_sync.py sync --data-dir <目录>")
        return

    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)

    size_mb = status.get("storage_bytes", 0) / (1024 * 1024)
    print(f"🧠 外脑状态：")
    print(f"  上次同步: {status.get('last_sync', '未知')}")
    print(f"  文档总数: {status.get('total_pages', 0)} 篇")
    print(f"  文本块数: {status.get('total_chunks', 0)} 块")
    print(f"  向量维度: {status.get('vector_dimensions', 0)}")
    print(f"  存储大小: {size_mb:.1f} MB")
    print(f"  Embedding: {status.get('embedding_provider', '?')} ({status.get('embedding_model', '?')})")


# ─────────────────────────── 主入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Notion Knowledge Base Sync Tool - OpenClaw External Brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 初始化配置（Azure）
  python notion_sync.py init-config \\
    --notion-token "ntn_xxx" \\
    --notion-db-id "abc123" \\
    --embedding-provider azure \\
    --embedding-api-key "your-key" \\
    --azure-endpoint "https://xxx.openai.azure.com" \\
    --data-dir "./data"

  # 初始化配置（OpenAI）
  python notion_sync.py init-config \\
    --notion-token "ntn_xxx" \\
    --notion-db-id "abc123" \\
    --embedding-provider openai \\
    --embedding-api-key "sk-xxx" \\
    --data-dir "./data"

  # 初始化配置（Gemini）
  python notion_sync.py init-config \\
    --notion-token "ntn_xxx" \\
    --notion-db-id "abc123" \\
    --embedding-provider gemini \\
    --embedding-api-key "AIzaXxx" \\
    --data-dir "./data"

  # 执行同步
  python notion_sync.py sync --data-dir "./data"

  # 查看状态
  python notion_sync.py status --data-dir "./data"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init-config
    p_init = subparsers.add_parser("init-config", help="初始化/更新配置")
    p_init.add_argument("--notion-token", help="Notion Integration Token")
    p_init.add_argument("--notion-db-id", help="Notion 数据库 ID")
    p_init.add_argument("--embedding-provider",
                        choices=["azure", "openai", "gemini"],
                        help="Embedding API 提供商")
    p_init.add_argument("--embedding-api-key", help="Embedding API Key")
    p_init.add_argument("--azure-endpoint", help="Azure OpenAI Endpoint URL")
    p_init.add_argument("--azure-deployment",
                        default="text-embedding-3-small",
                        help="Azure 部署名称（默认 text-embedding-3-small）")
    p_init.add_argument("--data-dir", required=True, help="数据目录路径")
    p_init.add_argument("--update-only", action="store_true",
                        help="仅更新指定字段，保留其他配置")
    p_init.add_argument("--append-db", action="store_true",
                        help="追加新的 Notion 数据库 ID")

    # sync
    p_sync = subparsers.add_parser("sync", help="执行同步")
    p_sync.add_argument("--data-dir", required=True, help="数据目录路径")
    p_sync.add_argument("--full", action="store_true",
                        help="强制全量同步（忽略增量缓存）")

    # status
    p_status = subparsers.add_parser("status", help="查看同步状态")
    p_status.add_argument("--data-dir", required=True, help="数据目录路径")

    args = parser.parse_args()

    if args.command == "init-config":
        cmd_init_config(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
