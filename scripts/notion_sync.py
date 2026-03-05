#!/usr/bin/env python3
"""
notion_sync.py — Notion 知识库夜间同步脚本
功能：
  init-config  初始化/更新 API 配置
  sync         拉取 Notion 数据库 → 切块 → 向量化 → 存本地
  status       查看同步状态
"""

import argparse
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

from embedding_utils import (
    encode_key, decode_key, load_config as _load_config, save_config,
    make_request_with_retry as _make_request_with_retry,
    get_embeddings, test_embedding,
)


# ─────────────────────────── 常量 ───────────────────────────

CHUNK_SIZE = 500       # 每个文本块的最大字符数
CHUNK_OVERLAP = 50     # 块间重叠字符数
NOTION_API_VERSION = "2025-09-03"
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

def load_config(data_dir: str) -> dict:
    """加载配置文件（带错误退出）"""
    config = _load_config(data_dir)
    if config is None:
        print("❌ 配置文件不存在，请先运行 init-config")
        sys.exit(1)
    return config


# ─────────────────────────── Notion API ───────────────────────────

def notion_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }


def fetch_notion_data_source(token: str, data_source_id: str) -> list:
    """获取数据源中的所有页面"""
    headers = notion_headers(token)
    url = f"{NOTION_BASE_URL}/data_sources/{data_source_id}/query"
    pages = []
    payload = {"page_size": 100}
    has_more = True

    while has_more:
        resp = _make_request_with_retry("post", url, headers, json_data=payload)
        if not resp:
            raise RuntimeError(f"Notion 数据源 {data_source_id} 拉取失败: 网络请求无响应")
        if resp.status_code != 200:
            raise RuntimeError(
                f"Notion API 错误 [{resp.status_code}] (data_source={data_source_id}): {resp.text[:200]}"
            )
        data = resp.json()
        pages.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        if has_more:
            payload["start_cursor"] = data["next_cursor"]

    return pages


def resolve_data_sources(token: str, config: dict, data_dir: str) -> list:
    """
    解析可同步的数据源。
    优先使用配置中的 notion_data_source_ids；若只有 notion_db_ids，则自动发现并缓存 data_source_id。
    """
    headers = notion_headers(token)
    configured_ds_ids = config.get("notion_data_source_ids", [])
    db_ids = config.get("notion_db_ids", [])

    sources = []
    seen_ids = set()

    def _add_source(ds_id: str, db_id: str = "", ds_name: str = ""):
        if not ds_id or ds_id in seen_ids:
            return
        seen_ids.add(ds_id)
        sources.append({
            "data_source_id": ds_id,
            "database_id": db_id,
            "name": ds_name,
        })

    # 1) 先保留显式配置的数据源 ID
    for ds_id in configured_ds_ids:
        _add_source(ds_id)

    # 2) 再从数据库发现数据源（兼容旧配置）
    discovered_ds_ids = []
    for db_id in db_ids:
        url = f"{NOTION_BASE_URL}/databases/{db_id}"
        resp = _make_request_with_retry("get", url, headers)
        if not resp:
            print(f"  ⚠️ 发现 data source 失败（网络无响应）: db={db_id}")
            continue
        if resp.status_code != 200:
            print(f"  ⚠️ 发现 data source 失败: db={db_id} [{resp.status_code}]")
            continue

        data = resp.json()
        for ds in data.get("data_sources", []):
            ds_id = ds.get("id", "")
            ds_name = ds.get("name", "")
            if ds_id:
                discovered_ds_ids.append(ds_id)
                _add_source(ds_id, db_id=db_id, ds_name=ds_name)

    # 3) 持久化发现结果，方便后续直接使用 data_source_id
    merged_ds_ids = list(dict.fromkeys(configured_ds_ids + discovered_ds_ids))
    if merged_ds_ids and merged_ds_ids != configured_ds_ids:
        config["notion_data_source_ids"] = merged_ds_ids
        save_config(data_dir, config)
        print(f"[migration] 已缓存 {len(merged_ds_ids)} 个 data_source_id 到配置")

    return sources


def fetch_page_content(token: str, page_id: str) -> str:
    """获取单个页面的全部文本内容（递归读取嵌套 block）"""
    headers = notion_headers(token)
    texts = []

    max_pages = 20
    page_count = 0

    def _walk_children(block_id: str, depth: int = 0):
        nonlocal page_count
        if depth > 30:
            return

        next_cursor = None
        while page_count < max_pages:
            page_count += 1
            url = f"{NOTION_BASE_URL}/blocks/{block_id}/children?page_size=100"
            if next_cursor:
                url += f"&start_cursor={next_cursor}"

            resp = _make_request_with_retry("get", url, headers)
            if not resp:
                raise RuntimeError(f"读取页面 {page_id} block 失败: 网络请求无响应")
            if resp.status_code != 200:
                raise RuntimeError(
                    f"读取页面 {page_id} block 失败 [{resp.status_code}]: {resp.text[:200]}"
                )

            data = resp.json()
            for block in data.get("results", []):
                text = extract_block_text(block)
                if text:
                    texts.append(text)
                if block.get("has_children") and block.get("id"):
                    _walk_children(block["id"], depth + 1)

            if data.get("has_more") and data.get("next_cursor"):
                next_cursor = data["next_cursor"]
            else:
                break

    _walk_children(page_id)

    if page_count >= max_pages:
        print(f"  ⚠️ 达到最大 block 分页限制(2000 blocks)，停止拉取页面 {page_id}")

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


# ─────────────────────── Embedding API（来自 embedding_utils.py）──────────────
# get_embeddings / test_embedding 已从 embedding_utils 导入


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
        if not config_path.exists():
            print("❌ 未找到现有配置。请先运行完整 init-config，再使用 --append-db")
            sys.exit(1)

        required = ["notion_token", "embedding_provider", "embedding_api_key"]
        missing = [r for r in required if r not in config or not config[r]]
        has_any_source = bool(config.get("notion_db_ids") or config.get("notion_data_source_ids"))
        if not has_any_source:
            missing.append("notion_db_ids/notion_data_source_ids")
        if missing:
            print(f"❌ 现有配置不完整，无法追加数据库，缺少: {missing}")
            sys.exit(1)

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
        elif args.notion_data_source_id:
            ds_ids = config.get("notion_data_source_ids", [])
            if args.notion_data_source_id not in ds_ids:
                ds_ids.append(args.notion_data_source_id)
                config["notion_data_source_ids"] = ds_ids
                save_config(data_dir, config)
                print(f"✅ 已添加 Notion 数据源: {args.notion_data_source_id}")
                print(f"   当前数据源列表: {ds_ids}")
            else:
                print(f"ℹ️ 数据源 {args.notion_data_source_id} 已存在")
        else:
            print("❌ --append-db 需要提供 --notion-db-id 或 --notion-data-source-id")
        return

    # 正常初始化 / 更新
    if args.notion_token:
        config["notion_token"] = encode_key(args.notion_token)
    if args.notion_db_id:
        existing_ids = config.get("notion_db_ids", [])
        if args.notion_db_id not in existing_ids:
            existing_ids.append(args.notion_db_id)
        config["notion_db_ids"] = existing_ids
    if args.notion_data_source_id:
        existing_ids = config.get("notion_data_source_ids", [])
        if args.notion_data_source_id not in existing_ids:
            existing_ids.append(args.notion_data_source_id)
        config["notion_data_source_ids"] = existing_ids
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
    required = ["notion_token", "embedding_provider", "embedding_api_key"]
    missing = [r for r in required if r not in config or not config[r]]
    has_any_source = bool(config.get("notion_db_ids") or config.get("notion_data_source_ids"))
    if not has_any_source:
        missing.append("notion_db_ids/notion_data_source_ids")
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
    for db_id in config.get("notion_db_ids", []):
        try:
            resp = requests.get(
                f"{NOTION_BASE_URL}/databases/{db_id}",
                headers=notion_headers(token),
                timeout=10
            )
            if resp.status_code == 200:
                db_obj = resp.json()
                title = db_obj.get("title", [{}])
                db_name = title[0].get("plain_text", "无标题") if title else "无标题"
                ds_count = len(db_obj.get("data_sources", []))
                print(f"  ✅ 数据库连接成功: {db_name} (data_sources={ds_count})")
            else:
                print(f"  ❌ 数据库 {db_id} 连接失败 [{resp.status_code}]")
        except Exception as e:
            print(f"  ❌ 数据库 {db_id} 连接异常: {e}")

    for ds_id in config.get("notion_data_source_ids", []):
        try:
            resp = requests.get(
                f"{NOTION_BASE_URL}/data_sources/{ds_id}",
                headers=notion_headers(token),
                timeout=10
            )
            if resp.status_code == 200:
                print(f"  ✅ 数据源连接成功: {ds_id}")
            else:
                print(f"  ❌ 数据源 {ds_id} 连接失败 [{resp.status_code}]")
        except Exception as e:
            print(f"  ❌ 数据源 {ds_id} 连接异常: {e}")

    print("\n🔗 验证 Embedding API...")
    test_embedding(config)


# ─────────────────────────── 增量同步辅助 ───────────────────────────

def load_existing_sync_data(brain_dir: Path):
    """加载已有的向量和元数据，用于增量同步"""
    vectors_path = brain_dir / "vectors.npy"
    meta_path = brain_dir / "chunks_meta.json"

    if not vectors_path.exists() or not meta_path.exists():
        return None, None

    vectors = np.load(str(vectors_path), mmap_mode="r")
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
    sources = resolve_data_sources(token, config, data_dir)
    full_sync = getattr(args, 'full', False)

    if not sources:
        print("Error: no Notion data sources configured or discoverable")
        sys.exit(1)

    brain_dir = Path(data_dir) / "my_brain_db"
    brain_dir.mkdir(parents=True, exist_ok=True)

    # ── 文件锁防止并发同步 ──
    lock_path = brain_dir / "sync.lock"
    lock_file = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("❌ 另一个同步进程正在运行，请稍后重试")
        lock_file.close()
        sys.exit(1)

    try:
        _do_sync(data_dir, config, token, sources, full_sync, brain_dir)
    except Exception as e:
        print(f"❌ 同步失败，已终止写入: {e}")
        sys.exit(1)
    finally:
        lock_file.close()
        try:
            lock_path.unlink()
        except OSError:
            pass


def _do_sync(data_dir, config, token, sources, full_sync, brain_dir):
    """实际的同步逻辑（在文件锁保护下运行）"""
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
    final_meta = []         # 最终的元数据列表
    final_vectors_parts = []  # 最终的向量列表（待合并）

    new_chunks_to_embed = []   # 需要新向量化的文本
    new_meta_to_embed = []     # 对应的元数据

    stats = {"unchanged": 0, "modified": 0, "new": 0, "deleted": 0}
    seen_page_ids = set()

    print(f"syncing {len(sources)} data source(s)...\n")

    for source in sources:
        ds_id = source["data_source_id"]
        db_id = source.get("database_id", "")
        source_label = source.get("name") or ds_id

        print(f"  data_source: {source_label} ({ds_id})")
        pages = fetch_notion_data_source(token, ds_id)
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
                    "source_data_source": ds_id,
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

    # ── 释放内存映射文件的句柄，准备覆盖写入 ──
    if not full_sync and old_vectors is not None:
        del old_vectors
        import gc
        gc.collect()

    # ── 保存（原子写入：先写临时文件，再替换）──
    tmp_vectors = brain_dir / "vectors_tmp.npy"
    tmp_meta = brain_dir / "chunks_meta_tmp.json"
    np.save(str(tmp_vectors), all_vectors)
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, ensure_ascii=False, indent=2)
    # 替换正式文件（os.replace 在同一文件系统上是原子操作）
    os.replace(str(tmp_vectors), str(vectors_path))
    os.replace(str(tmp_meta), str(meta_path))

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

    # ── 清理备份（所有文件写入成功后再删除）──
    backup_path = brain_dir / "vectors_backup.npy"
    try:
        # 验证刚写入的文件完整性
        _test_vectors = np.load(str(vectors_path), mmap_mode="r")
        assert _test_vectors.shape[0] == len(final_meta), "向量行数与元数据不匹配"
        del _test_vectors
        if backup_path.exists():
            backup_path.unlink()
    except Exception as e:
        print(f"  ⚠️ 写入验证失败，保留备份文件: {e}")

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
        --notion-data-source-id "def456" \\
    --embedding-provider azure \\
    --embedding-api-key "your-key" \\
    --azure-endpoint "https://xxx.openai.azure.com" \\
    --data-dir "./data"

  # 初始化配置（OpenAI）
  python notion_sync.py init-config \\
    --notion-token "ntn_xxx" \\
    --notion-db-id "abc123" \\
        --notion-data-source-id "def456" \\
    --embedding-provider openai \\
    --embedding-api-key "sk-xxx" \\
    --data-dir "./data"

  # 初始化配置（Gemini）
  python notion_sync.py init-config \\
    --notion-token "ntn_xxx" \\
    --notion-db-id "abc123" \\
        --notion-data-source-id "def456" \\
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
    p_init.add_argument("--notion-data-source-id", help="Notion 数据源 ID")
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
