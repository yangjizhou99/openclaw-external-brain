# \ud83d\udee0\ufe0f OpenClaw \u670d\u52a1\u5668\u90e8\u7f72\u6307\u5357

\u672c Skill \u76f4\u63a5\u90e8\u7f72\u5728 OpenClaw \u6240\u5728\u7684\u4e91\u670d\u52a1\u5668\u4e0a\uff0c\u4e0d\u9700\u8981\u989d\u5916\u670d\u52a1\u5668\u3002

---

## \u7b2c\u4e00\u6b65\uff1a\u5f00\u542f Swap\uff082GB \u8fd0\u5b58\u5fc5\u505a\uff09

\u5982\u679c\u4f60\u7684\u670d\u52a1\u5668\u53ea\u6709 2GB \u8fd0\u5b58\uff0c\u5f3a\u70c8\u5efa\u8bae\u5f00\u542f Swap \u9632\u6b62 OOM\uff1a

```bash
# 1. \u521b\u5efa 2GB swap \u6587\u4ef6
sudo fallocate -l 2G /swapfile

# 2. \u8bbe\u7f6e\u6743\u9650
sudo chmod 600 /swapfile

# 3. \u683c\u5f0f\u5316
sudo mkswap /swapfile

# 4. \u542f\u7528
sudo swapon /swapfile

# 5. \u9a8c\u8bc1\uff08\u5e94\u663e\u793a 2G swap\uff09
free -h

# 6. \u5f00\u673a\u81ea\u52a8\u6302\u8f7d
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## \u7b2c\u4e8c\u6b65\uff1a\u5b89\u88c5\u4f9d\u8d56

\u5728 OpenClaw \u7684 Python \u73af\u5883\u4e2d\u5b89\u88c5\uff1a

```bash
pip install numpy requests
```

> \u4ec5 2 \u4e2a\u5305\uff0c\u65e0\u989d\u5916\u8d1f\u62c5\u3002

---

## \u7b2c\u4e09\u6b65\uff1a\u914d\u7f6e Notion Integration

1. \u8bbf\u95ee https://www.notion.so/my-integrations
2. \u521b\u5efa New Integration\uff0c\u8bb0\u4e0b Token\uff08`ntn_` \u5f00\u5934\uff09
3. \u5728\u76ee\u6807 Notion \u6570\u636e\u5e93 \u2192 `...` \u2192 **Connections** \u2192 \u6dfb\u52a0\u8be5 Integration
4. \u4ece\u6570\u636e\u5e93 URL \u590d\u5236\u6570\u636e\u5e93 ID

---

## \u7b2c\u56db\u6b65\uff1a\u521d\u59cb\u5316 & \u9996\u6b21\u540c\u6b65

```bash
# \u914d\u7f6e\uff08Gemini API Key \u5df2\u6709\uff09
python scripts/notion_sync.py init-config \
  --notion-token "ntn_\u4f60\u7684token" \
  --notion-db-id "\u4f60\u7684\u6570\u636e\u5e93ID" \
  --embedding-provider gemini \
  --embedding-api-key "\u4f60\u7684Gemini Key" \
  --data-dir "./data"

# \u9996\u6b21\u540c\u6b65
python scripts/notion_sync.py sync --data-dir "./data"

# \u6d4b\u8bd5\u68c0\u7d22
python scripts/tool_search_brain.py search \
  --query "\u6d4b\u8bd5\u4e00\u4e0b" \
  --data-dir "./data"
```

---

## \u7b2c\u4e94\u6b65\uff1a\u914d\u7f6e Cron \u5b9a\u65f6\u540c\u6b65

```bash
crontab -e

# \u6bcf\u5929\u51cc\u6668 3:00 \u540c\u6b65\uff0c\u907f\u514d\u4e0e OpenClaw \u4e3b\u670d\u52a1\u62a2\u5185\u5b58
0 3 * * * cd /path/to/openclaw-external-brain && python scripts/notion_sync.py sync --data-dir "./data" >> ./data/sync.log 2>&1
```

---

## \u5185\u5b58\u9884\u4f30

| \u7ec4\u4ef6 | \u5185\u5b58\u5360\u7528 |
|------|----------|
| OpenClaw \u4e3b\u670d\u52a1 | \u6b63\u5e38\u8fd0\u884c |
| \u540c\u6b65\u811a\u672c\uff08\u51cc\u6668\uff09 | ~50-100 MB\uff0c\u8fd0\u884c\u5b8c\u9000\u51fa |
| \u68c0\u7d22\u811a\u672c\uff08\u968f\u65f6\uff09 | ~10 MB\uff0c\u5355\u6b21\u8c03\u7528 |

> \u540c\u6b65\u548c\u68c0\u7d22\u811a\u672c\u90fd\u662f\u72ec\u7acb\u8fdb\u7a0b\uff0c\u4e0d\u5e38\u9a7b\u5185\u5b58\u30022GB + 2GB Swap \u7eB0\u7eB0\u6709\u4f59\u3002

---

## \u6545\u969c\u6392\u67e5

| \u95ee\u9898 | \u6392\u67e5\u65b9\u6cd5 |
|------|----------|
| \u8fdb\u7a0b\u88ab Killed | `dmesg | grep -i oom`\uff0c\u68c0\u67e5 Swap |
| Notion API 403 | \u68c0\u67e5 Integration \u662f\u5426\u6dfb\u52a0\u5230 Connections |
| Gemini 404 | \u68c0\u67e5\u6a21\u578b\u540d\u662f\u5426\u4e3a `gemini-embedding-001` |
| \u5411\u91cf\u7ef4\u5ea6\u4e0d\u5339\u914d | \u66f4\u6362\u63d0\u4f9b\u5546\u540e\u91cd\u65b0 `sync --full` |
