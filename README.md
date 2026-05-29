# Prompt Council — 多智能体交叉验证系统

三个 AI 模型组成审议委员会，通过 Maker → Challenger → Judge 的辩论流程，将你的原始需求精炼为高质量 Prompt。

## 架构

```
用户需求
  │
  ▼
┌─────────────────────────────────────┐
│  Round 1                           │
│  🔨 Maker (Gemini 2.5 Flash)  → 起草  │
│  ⚔️ Challenger (Qwen 2.5)     → 审计  │
├─────────────────────────────────────┤
│  Round 2                           │
│  🔨 Maker (Gemini 2.5 Flash)  → 修订  │
│  ⚔️ Challenger (Qwen 2.5)     → 复核  │
├─────────────────────────────────────┤
│  ⚖️ Judge (DeepSeek V4 Pro)   → 终审  │
└─────────────────────────────────────┘
  │
  ▼
最终 Prompt + 保存到 outputs/
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.template .env
# 编辑 .env，填入:
#   GEMINI_API_KEY=...
#   SILICONFLOW_API_KEY=...
#   DEEPSEEK_API_KEY=...

# 3. 启动
python main.py
```

## 交互命令

| 命令 | 作用 |
|---|---|
| 输入需求文本 | 启动一轮完整的 Maker → Challenger → Judge 辩论 |
| `history` | 查看当前会话所有历史记录 |
| `stats` | 查看各模型 API 累计调用次数 |
| `quit` / `exit` | 退出 |

## 数据格式

**Maker 输出:**
```json
{"draft": "草稿内容", "rationale": "设计思路"}
```

**Challenger 输出:**
```json
{"issues": ["漏洞1", "漏洞2"], "severity": "High/Medium/Low", "improvement_suggestions": "修改建议"}
```

**Judge 输出:** 纯文本 Prompt（可直接复制使用）

## 容错机制

| 机制 | 说明 |
|---|---|
| 指数退避重试 | 429/5xx 错误自动重试，间隔 2s → 4s → 8s → ... → 60s，最多 5 次 |
| JSON 强制提取 | 3 层策略剥离 markdown 标记，确保解析不崩溃 |
| Bug 熔断 | 同一环节连续失败 3 次，自动挂起 |
| 循环熔断 | 单次会话最多 5 轮辩论 |
| 资金熔断 | 检测到 401/403/计费错误立即停止 |
| 调用间隔 | 每两次 API 调用之间 1.5 秒延迟，防 429 封禁 |

## 配置

所有可调参数在 `.env` 中通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_RETRIES` | 5 | 最大重试次数 |
| `BASE_DELAY` | 2.0 | 初始退避延迟 (秒) |
| `MAX_DELAY` | 60.0 | 退避上限 (秒) |
| `MAX_DEBATE_ROUNDS` | 2 | 辩论轮数 |
| `INTER_CALL_DELAY` | 1.5 | API 调用间隔 (秒) |

## 项目结构

```
├── main.py           # 控制流 + 交互终端
├── llm_client.py     # LLM 客户端 (重试 / JSON 提取)
├── roles.py          # System Prompt 定义
├── config.py         # 配置读取
├── requirements.txt
├── .env.template
└── outputs/          # 结果自动保存目录
```

## 输出

每轮辩论结果自动追加到 `outputs/final_results.md`，包含用户需求、最终 Prompt、辩论历史和 API 调用统计。
