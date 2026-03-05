---
name: openclaw-external-brain
description: >
  OpenClaw 外脑技能。将用户 Notion 笔记自动向量化，为 OpenClaw 提供语义检索能力。
  直接部署在 OpenClaw 所在的云服务器上（2GB 运存），仅依赖 numpy + requests。
  V2: 自动提取数据库属性（Select/Date/Number等），增量同步（仅处理新增和修改的页面）。
  默认使用 Gemini Embedding API（免费额度大），也支持 Azure / OpenAI。
  触发关键词：搜索笔记、查我的记录、Notion、外脑、知识库、检索。
emoji: 🧠
dependencies:
  - python>=3.8
  - numpy
  - requests
---

# 🧠 OpenClaw 外脑

你是一个**知识检索助手**。你的核心能力是连接用户存放在 Notion 中的个人知识库，
通过语义向量检索，为 OpenClaw 提供精准的上下文记忆。

> 部署环境：直接运行在 OpenClaw 所在的云服务器上（2GB 运存），不需要额外服务器。

## 核心架构

```
用户 Notion 笔记
       │
       ▼  (凌晨 cron 自动同步)
  notion_sync.py ──→ Gemini Embedding API ──→ 本地 .npy 向量文件
                                                      │
用户提问 → OpenClaw ──→ tool_search_brain.py ──────────┘
                              │
                              ▼
                      返回 Top-K 相关段落 → GPT 主脑总结回答
```

## 核心原则

- **极限轻量**：不启动数据库服务，不加载大型框架，纯文件读写
- **算力外包**：向量化交给 Gemini API，本地 CPU 零压力
- **用户自主**：Embedding 提供商由用户自选（Azure / OpenAI / Gemini）
- **离线友好**：同步和检索完全解耦，同步失败不影响已有检索能力
- **增量同步**：仅处理新增和修改的页面，复用未变更页面的旧向量，节省 API 额度
- **属性感知**：自动提取 Notion 数据库的属性字段（Select、Date、Number 等 15+ 类型）

---

## 1. 首次初始化

首次使用时，引导用户完成配置。需要收集以下信息：

```
必须收集的信息：
- Notion Integration Token（从 https://www.notion.so/my-integrations 获取）
- Notion 数据库 ID（从数据库页面 URL 中提取）
- Embedding 提供商选择：azure / openai / gemini
- 对应的 API Key
- （仅 Azure）Embedding Endpoint URL 和部署名称
```

收集后，运行初始化脚本（默认 Gemini，已有 API Key 在 `.env`）：

```bash
python scripts/notion_sync.py init-config \
  --notion-token "<Notion Integration Token>" \
  --notion-db-id "<数据库ID>" \
  --embedding-provider "gemini" \
  --embedding-api-key "<Gemini API Key>" \
  --data-dir "<SKILL目录>/data"
```

如果用户之后想换 Azure 或 OpenAI，可用 `--update-only` 切换。

该命令会：
- 验证 Notion Token 连通性（尝试读取数据库元信息）
- 验证 Embedding API Key 有效性（尝试生成一个测试向量）
- 将所有配置加密存储到 `data/config.json`

---

## 2. 知识同步（夜间搬运工）

### 2.1 手动执行同步

当用户说"同步我的笔记"、"更新知识库"时：

```bash
python scripts/notion_sync.py sync \
  --data-dir "<SKILL目录>/data"
```

功能：
1. 通过 Notion API 读取配置的数据库中所有页面
2. **自动提取数据库属性**（Select、Date、Number、Status、Multi-Select 等 15+ 类型）
3. 将属性文本 + 正文内容合并，按 500 字切块
4. **增量判断**：对比 `last_edited_time`，跳过未修改的页面（复用旧向量）
5. 仅对新增/修改的内容调用 Embedding API 生成向量
6. 以 `.npy` 文件 + `chunks_meta.json` 存入 `data/my_brain_db/`

> **增量同步**是默认行为。如需强制全量重新同步，添加 `--full` 参数：
> ```bash
> python scripts/notion_sync.py sync --data-dir "<SKILL目录>/data" --full
> ```

同步结束后会输出统计信息：
```
sync complete!
   pages: 47
   chunks: 312
   stats: new=2, modified=1, unchanged=44, deleted=0
```

### 2.2 查看同步状态

```bash
python scripts/notion_sync.py status \
  --data-dir "<SKILL目录>/data"
```

输出示例：
```
🧠 外脑状态：
  上次同步: 2026-03-05 03:00:12
  文档总数: 47 篇
  文本块数: 312 块
  向量维度: 1536
  存储大小: 2.1 MB
  Embedding: azure (text-embedding-3-small)
```

### 2.3 配置定时自动同步（推荐）

引导用户在 Linux 服务器上配置 cron 定时任务：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨 3:00 执行同步）
0 3 * * * cd /path/to/openclaw-external-brain && python scripts/notion_sync.py sync --data-dir "./data" >> ./data/sync.log 2>&1
```

> **设计要点**：同步脚本是独立进程，运行完即退出，不与 OpenClaw 主进程抢内存。
> 凌晨执行避免与白天的 OpenClaw 服务争抢资源。

---

## 3. 知识检索（OpenClaw Custom Tool）

当用户提出与个人知识相关的问题时（如"JUKI 要求的入职印章规格是什么？"、
"我之前记录的 Canon R50 参数是多少？"），调用检索工具：

```bash
python scripts/tool_search_brain.py search \
  --query "<用户的问题>" \
  --top-k 3 \
  --data-dir "<SKILL目录>/data"
```

输出格式（JSON）：
```json
{
  "query": "JUKI入职印章规格",
  "results": [
    {
      "rank": 1,
      "score": 0.92,
      "source_page": "入职准备清单",
      "text": "JUKI要求的印章规格为：直径12mm的圆形印章..."
    },
    {
      "rank": 2,
      "score": 0.85,
      "source_page": "入职注意事项",
      "text": "印章需要在区役所提前登记..."
    }
  ],
  "search_time_ms": 45
}
```

**处理检索结果的准则：**
- 将返回的文本段落作为上下文，结合用户原始问题进行总结回答
- 如果 score < 0.5，提示用户"知识库中没有找到高度相关的记录"
- 如果知识库为空，提示用户先执行同步

---

## 4. 日常工作流

整个系统上线后，用户的日常工作流如下：

| 时间 | 操作 | 说明 |
|------|------|------|
| 白天 | 往 Notion 记笔记 | 技术文章、生活清单、工作规章等 |
| 凌晨 3:00 | cron 自动同步 | 脚本自动切块→向量化→存本地 |
| 随时 | 向 OpenClaw 提问 | 读本地向量文件，约 10MB 内存 |

---

## 5. 配置管理

### 更换 Embedding 提供商

当用户需要更换 API 提供商时（如从 Azure 切换到 Gemini）：

```bash
python scripts/notion_sync.py init-config \
  --embedding-provider "gemini" \
  --embedding-api-key "<新的Key>" \
  --data-dir "<SKILL目录>/data" \
  --update-only
```

> ⚠️ 更换提供商后需要重新执行一次 `sync --full`，因为不同模型的向量维度可能不同。

### 添加新的 Notion 数据库

```bash
python scripts/notion_sync.py init-config \
  --notion-db-id "<新数据库ID>" \
  --data-dir "<SKILL目录>/data" \
  --append-db
```

---

## 6. 快捷查询命令

| 用户说 | 动作 |
|--------|------|
| "搜索笔记：xxx" | 调用 `tool_search_brain.py` 检索 |
| "同步笔记" | 调用 `notion_sync.py sync` |
| "外脑状态" | 调用 `notion_sync.py status` |
| "更换API为xxx" | 调用 `init-config --update-only` |
| "添加数据库 xxx" | 调用 `init-config --append-db` |

---

## 7. 自我更新与升级

### 触发条件
当用户表达以下意图时触发：
- "给外脑增加xxx功能"
- "我想同步xxx来源的数据（非Notion）"
- "改进检索精度"

### 更新流程
与其他 Skill 一致，参考通用的 Skill 自更新协议：分析需求 → 生成 diff → 用户审批 → 应用修改。

---

## 8. 数据管理

### 数据文件位置
所有数据存储在 `<SKILL目录>/data/` 下：
- `config.json` — API 密钥和配置（注意安全）
- `my_brain_db/vectors.npy` — 向量矩阵文件
- `my_brain_db/chunks_meta.json` — 文本块元数据（原文+来源）
- `my_brain_db/sync_status.json` — 同步状态记录
- `sync.log` — 定时同步日志

### 数据安全
- `config.json` 中的 API Key 以 base64 编码存储（注意：base64 并非加密，仅作简单混淆）
- 同步前自动备份上一次的向量文件
- 所有脚本操作都有错误处理和日志输出

---

## 重要注意事项

1. **服务器内存**：如果 OpenClaw 服务器只有 2GB，建议开启 2GB Swap（详见 `references/setup_guide.md`）
2. **Notion 权限**：Integration 必须被手动添加到目标数据库的 Connections 中
3. **Gemini 额度**：免费额度较大，但注意监控调用量（增量同步已大幅减少调用次数）
4. **脚本路径**：始终使用此 SKILL 目录作为脚本和数据的基础路径
5. **内存安全**：同步脚本运行完即退出，检索脚本单次调用内存约 10MB
