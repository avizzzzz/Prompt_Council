#!/usr/bin/env python3
"""Multi-agent collaborative cross-validation system — main control flow.

Flow per turn:
  1. User submits a request
  2. Maker drafts a prompt
  3. Challenger audits the draft
  4. Maker revises based on feedback
  5. Challenger re-audits (final pass)
  6. Judge synthesizes the final prompt

Circuit breakers:
  - Same-error fuse: 3 consecutive failures on the same step → abort task
  - Loop fuse: max 5 auto-loops per session
  - Cost fuse: detect non-recoverable API errors (401, 403) → abort immediately
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import INTER_CALL_DELAY, MAX_DEBATE_ROUNDS, OUTPUT_FILE
from llm_client import LLMClient
from roles import CHALLENGER_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, MAKER_SYSTEM_PROMPT

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,  # keep console clean; WARNING+ only
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("council")

# ── Global state ──────────────────────────────────────────────────────

chat_history: List[Dict[str, Any]] = []
client = LLMClient()

# Circuit breaker state
_consecutive_failures: Dict[str, int] = {}
_loop_count: int = 0
MAX_FAILURES = 3
MAX_LOOPS = 5


# ── Helpers ───────────────────────────────────────────────────────────

def _emoji(icon: str, label: str, *args: Any) -> None:
    """Print an emoji-prefixed status line."""
    msg = " ".join(str(a) for a in args)
    print(f"{icon}  [{label}] {msg}")


def _delay(seconds: float = INTER_CALL_DELAY) -> None:
    """Inter-call delay to avoid 429 rate-limiting."""
    if seconds > 0:
        time.sleep(seconds)


def _reset_circuit() -> None:
    global _consecutive_failures
    _consecutive_failures = {}


def _register_failure(step: str) -> None:
    global _consecutive_failures
    _consecutive_failures[step] = _consecutive_failures.get(step, 0) + 1


def _check_circuit() -> None:
    """Raise RuntimeError if any step has failed 3 consecutive times."""
    for step, count in _consecutive_failures.items():
        if count >= MAX_FAILURES:
            raise RuntimeError(
                f"🔴 熔断触发: '{step}' 连续失败 {count} 次，已挂起。请检查 API Key / 网络后重试。"
            )


def _check_cost_fuse(exception: Exception) -> None:
    """If the exception smells like a non-recoverable API error (401, 403, billing), abort."""
    msg = str(exception).lower()
    fatal = False
    if "401" in msg or "unauthorized" in msg:
        fatal = True
    if "403" in msg or "forbidden" in msg:
        fatal = True
    if "billing" in msg or "insufficient" in msg or "balance" in msg:
        fatal = True
    if fatal:
        raise RuntimeError(
            "🔴 资金/权限熔断触发: API 返回不可恢复错误 (401/403/计费问题)。已停止一切操作。\n"
            f"原始错误: {exception}"
        ) from exception


def _save_result(user_request: str, final_prompt: str, history: List[Dict[str, Any]]) -> None:
    """Append the final result to outputs/final_results.md."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n")
        f.write(f"## Session — {timestamp}\n\n")
        f.write(f"**User Request:** {user_request}\n\n")
        f.write(f"### Final Prompt\n\n```\n{final_prompt}\n```\n\n")
        f.write(f"### Debate History\n\n")
        for i, entry in enumerate(history, 1):
            f.write(f"#### Round {i}\n\n")
            f.write(f"- **Maker draft:** {entry.get('maker_draft', 'N/A')[:300]}...\n")
            f.write(f"- **Challenger issues:** {entry.get('challenger_issues', [])}\n")
            f.write(f"- **Severity:** {entry.get('severity', 'N/A')}\n\n")
        f.write(f"### API Call Stats\n\n```json\n{json.dumps(client.stats, indent=2)}\n```\n")
        f.write(f"\n")


# ── Core loop ─────────────────────────────────────────────────────────

def run_debate(user_request: str) -> Optional[str]:
    """Run one full debate cycle for a user request. Returns final prompt or None."""
    global _loop_count, _consecutive_failures

    _loop_count += 1
    if _loop_count > MAX_LOOPS:
        print("\n⛔ 循环熔断: 已达到单次会话最大循环次数 (5)，请重启程序。")
        return None

    _reset_circuit()
    debate_rounds: List[Dict[str, Any]] = []

    print()
    _emoji("📝", "输入", f"用户需求: {user_request}")

    # ── Round 1: Maker → Challenger ───────────────────────────────────

    _emoji("🔨", "Maker", "第1轮 — 正在生成初稿...")
    _delay()

    try:
        maker_result = client.call_gemini(MAKER_SYSTEM_PROMPT, user_request)
    except Exception as e:
        _check_cost_fuse(e)
        _register_failure("Maker-R1")
        _check_circuit()
        _emoji("❌", "Maker", f"第1轮失败: {e}")
        return None

    draft_v1 = maker_result.get("draft", "")
    rationale_v1 = maker_result.get("rationale", "")
    _emoji("✅", "Maker", f"初稿完成 — 理由: {rationale_v1[:80]}...")
    print(f"   📄 草稿: {draft_v1[:200]}...")

    _delay()

    _emoji("⚔️", "Challenger", "第1轮 — 正在审计初稿...")
    challenger_input = f"User's original request:\n{user_request}\n\nMaker's draft prompt:\n{draft_v1}"

    try:
        challenger_result = client.call_qwen(CHALLENGER_SYSTEM_PROMPT, challenger_input)
    except Exception as e:
        _check_cost_fuse(e)
        _register_failure("Challenger-R1")
        _check_circuit()
        _emoji("❌", "Challenger", f"第1轮失败: {e}")
        return None

    issues_v1 = challenger_result.get("issues", [])
    severity_v1 = challenger_result.get("severity", "Unknown")
    improvements_v1 = challenger_result.get("improvement_suggestions", "")

    _emoji("🔍", "Challenger", f"发现 {len(issues_v1)} 个问题 | 严重程度: {severity_v1}")
    for issue in issues_v1:
        print(f"   ⚡ {issue}")

    debate_rounds.append({
        "round": 1,
        "maker_draft": draft_v1,
        "maker_rationale": rationale_v1,
        "challenger_issues": issues_v1,
        "severity": severity_v1,
        "improvements": improvements_v1,
    })

    _delay()

    # ── Round 2: Maker revises → Challenger re-audits ─────────────────

    _emoji("🔨", "Maker", "第2轮 — 根据反馈修订中...")
    revision_input = (
        f"Original user request:\n{user_request}\n\n"
        f"Your previous draft:\n{draft_v1}\n\n"
        f"Challenger found these issues:\n{json.dumps(issues_v1, indent=2)}\n"
        f"Severity: {severity_v1}\n"
        f"Suggested improvements: {improvements_v1}\n\n"
        f"Please produce a revised draft that addresses all valid criticisms."
    )

    try:
        maker_v2 = client.call_gemini(MAKER_SYSTEM_PROMPT, revision_input)
    except Exception as e:
        _check_cost_fuse(e)
        _register_failure("Maker-R2")
        _check_circuit()
        _emoji("❌", "Maker", f"第2轮失败: {e}")
        return None

    draft_v2 = maker_v2.get("draft", draft_v1)  # fall back to v1 if empty
    rationale_v2 = maker_v2.get("rationale", "")
    _emoji("✅", "Maker", f"修订完成 — 理由: {rationale_v2[:80]}...")
    print(f"   📄 修订稿: {draft_v2[:200]}...")

    _delay()

    _emoji("⚔️", "Challenger", "第2轮 — 最终审计中...")
    challenger_input_v2 = (
        f"User's original request:\n{user_request}\n\n"
        f"Maker's REVISED draft:\n{draft_v2}\n\n"
        f"(Previous issues found: {json.dumps(issues_v1)})"
    )

    try:
        challenger_v2 = client.call_qwen(CHALLENGER_SYSTEM_PROMPT, challenger_input_v2)
    except Exception as e:
        _check_cost_fuse(e)
        _register_failure("Challenger-R2")
        _check_circuit()
        _emoji("❌", "Challenger", f"第2轮失败: {e}")
        return None

    issues_v2 = challenger_v2.get("issues", [])
    severity_v2 = challenger_v2.get("severity", "Unknown")
    improvements_v2 = challenger_v2.get("improvement_suggestions", "")

    _emoji("🔍", "Challenger", f"第2轮 — {len(issues_v2)} 个遗留问题 | 严重程度: {severity_v2}")

    debate_rounds.append({
        "round": 2,
        "maker_draft": draft_v2,
        "maker_rationale": rationale_v2,
        "challenger_issues": issues_v2,
        "severity": severity_v2,
        "improvements": improvements_v2,
    })

    _delay()

    # ── Judge: final synthesis ────────────────────────────────────────

    _emoji("⚖️", "Judge", "终审中 — 综合所有辩论生成最终 Prompt...")

    judge_input_parts = [f"User request: {user_request}\n\n## Debate History\n"]
    for r in debate_rounds:
        judge_input_parts.append(
            f"### Round {r['round']}\n"
            f"Maker draft: {r['maker_draft']}\n"
            f"Challenger issues: {json.dumps(r['challenger_issues'])}\n"
            f"Severity: {r['severity']}\n"
            f"Improvements: {r['improvements']}\n"
        )
    judge_input = "\n".join(judge_input_parts)

    try:
        final_prompt = client.call_deepseek(JUDGE_SYSTEM_PROMPT, judge_input)
    except Exception as e:
        _check_cost_fuse(e)
        _register_failure("Judge")
        _check_circuit()
        _emoji("❌", "Judge", f"终审失败: {e}")
        return None

    _emoji("🏆", "结果", "最终 Prompt 已生成!")
    print(f"\n{'─' * 60}")
    print(final_prompt)
    print(f"{'─' * 60}\n")

    # ── Save ──────────────────────────────────────────────────────────

    _emoji("💾", "保存", f"写入 {OUTPUT_FILE} ...")
    try:
        _save_result(user_request, final_prompt, debate_rounds)
        _emoji("✅", "保存", "完成!")
    except Exception as e:
        _emoji("⚠️", "保存", f"写入文件失败: {e}")

    # Record in global chat history
    chat_history.append({
        "timestamp": datetime.now().isoformat(),
        "user_request": user_request,
        "debate_rounds": debate_rounds,
        "final_prompt": final_prompt,
    })

    _emoji("📊", "统计", f"API 调用: {json.dumps(client.stats)}")
    return final_prompt


# ── Interactive shell ─────────────────────────────────────────────────

def _validate_env() -> bool:
    """Check that required env vars are set. Print guidance if missing."""
    from config import DEEPSEEK_API_KEY, GEMINI_API_KEY, SILICONFLOW_API_KEY

    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not SILICONFLOW_API_KEY:
        missing.append("SILICONFLOW_API_KEY")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")

    if missing:
        print("⚠️  缺少以下 API Key (请在 .env 文件中配置):")
        for key in missing:
            print(f"   - {key}")
        print("\n💡 提示: 复制 .env.template 为 .env 并填入真实 Key 即可。")
        return False
    return True


def main() -> None:
    """Interactive multi-agent prompt council."""
    print("=" * 60)
    print("🏛️   多智能体交叉验证系统 — Prompt Council V3")
    print("=" * 60)
    print(f"📋 Maker:   Gemini 2.5 Flash")
    print(f"📋 Challenger: Qwen 2.5 (via SiliconFlow)")
    print(f"📋 Judge:   DeepSeek V4 Pro")
    print(f"🔄 辩论轮数: {MAX_DEBATE_ROUNDS} | ⏱️ 调用间隔: {INTER_CALL_DELAY}s")
    print(f"📁 结果保存: {OUTPUT_FILE}")
    print("=" * 60)

    if not _validate_env():
        print("\n⚠️  模拟模式启动 — 仅演示流程框架，无实际 API 调用。")
        print("   请配置 .env 后重启以启用完整功能。\n")
        # Don't exit — let the user play with the shell even without keys
        # (optional: could add a mock mode here)

    print("\n输入你的需求 (输入 'quit' 或 'exit' 退出, 'history' 查看历史):\n")

    while True:
        try:
            user_input = input("💬 你的需求 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见!")
            break

        if user_input.lower() == "history":
            if not chat_history:
                print("📭 暂无历史记录。")
            else:
                for i, entry in enumerate(chat_history, 1):
                    print(f"\n--- 记录 #{i} [{entry['timestamp']}] ---")
                    print(f"需求: {entry['user_request'][:100]}")
                    print(f"最终 Prompt: {entry['final_prompt'][:200]}...")
            continue

        if user_input.lower() == "stats":
            print(f"📊 API 累计调用: {json.dumps(client.stats, indent=2)}")
            continue

        # Run the debate
        try:
            run_debate(user_input)
        except RuntimeError as e:
            print(f"\n{e}")
            print("🛑 致命错误 — 程序终止。")
            break
        except Exception as e:
            print(f"\n❌ 未预期的错误: {e}")
            logger.exception("Unexpected error in run_debate")
            print("⚠️  继续运行 — 请输入下一个需求。")


if __name__ == "__main__":
    main()
