#!/usr/bin/env python3
"""
Step 1: Build compacted single-turn user messages from ATTS trajectories.

Compaction rules:
- Keep system/user messages and tool results (user-role outputs).
- Hide tool call syntax from the LLM (no `subagent(...)`, no `bash(...)`, etc.).
- Pair each `bash` command with its corresponding tool output.
- Stop before the commit tool call to avoid leaking the final answer.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compacted single-turn prompts")
    parser.add_argument("--atts-dir", required=True, type=Path, help="Path to ATTS rollout directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--question-ids",
        default="",
        help="Comma-separated list of question IDs to process (default: all)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (after filtering)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _find_traj_file(atts_dir: Path, qid: int) -> Path | None:
    candidate = atts_dir / f"question_{qid}" / f"{qid}.traj.json"
    if candidate.exists():
        return candidate
    fallback = atts_dir / f"question_{qid}" / "0.traj.json"
    if fallback.exists():
        return fallback
    return None


def _extract_tool_calls(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    extra = msg.get("extra", {})
    response = extra.get("response", {})
    choices = response.get("choices", [])
    if not choices:
        return []
    message = choices[0].get("message", {})
    return message.get("tool_calls", []) or []


def _parse_tool_call(tc: Dict[str, Any]) -> Tuple[str, str, str]:
    func = tc.get("function", {})
    name = func.get("name", "")
    args_raw = func.get("arguments", "{}")
    args: Dict[str, Any]
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}

    if args:
        key = next(iter(args.keys()))
        value = str(args[key])
    else:
        key = ""
        value = ""
    return name, key, value


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _normalize_tool_result(content: str) -> str:
    returncode = _extract_tag(content, "returncode")
    warning = _extract_tag(content, "warning")
    output = _extract_tag(content, "output")
    output_head = _extract_tag(content, "output_head")
    output_tail = _extract_tag(content, "output_tail")
    elided = _extract_tag(content, "elided_chars")

    if any([returncode, warning, output, output_head, output_tail, elided]):
        # For the natural prompt, keep the tool's textual output only (no XML tags, no return code noise).
        if output:
            return output.replace("<think>", "").replace("</think>", "").strip()
        if output_head or output_tail:
            parts: List[str] = []
            if warning:
                parts.append(f"[warning] {warning}")
            if output_head:
                parts.append("HEAD:\n" + output_head.replace("<think>", "").replace("</think>", "").strip())
            if elided:
                parts.append(str(elided).strip())
            if output_tail:
                parts.append("TAIL:\n" + output_tail.replace("<think>", "").replace("</think>", "").strip())
            return "\n\n".join(parts).strip()
        if warning:
            return warning.strip()
        return ""

    return content.strip()


def _extract_answer_stats_block(text: str) -> str | None:
    lines = [ln.rstrip() for ln in text.splitlines()]
    start_idx = None
    for i, line in enumerate(lines):
        if "Answer Statistics:" in line:
            start_idx = i
            break
    if start_idx is None:
        return None
    # Keep from Answer Statistics to end (or until a double blank if present)
    block = lines[start_idx:]
    return "\n".join(block).strip()


def _extract_problem(user_task_content: str) -> str:
    # Typical format starts with:
    #   ## Your Task
    #   <blank>
    #   <problem...>
    # and then "**Critical:**" or "## Step-by-Step Workflow"
    if "## Your Task" in user_task_content:
        after = user_task_content.split("## Your Task", 1)[1]
        after = after.lstrip()
        stop_markers = ["**Critical:**", "## Step-by-Step Workflow", "## Agent Count Guidelines"]
        stop_at = None
        for m in stop_markers:
            idx = after.find(m)
            if idx != -1:
                stop_at = idx if stop_at is None else min(stop_at, idx)
        if stop_at is not None:
            after = after[:stop_at]
        return after.strip()
    return user_task_content.strip()


def _render_bash_blocks(bash_runs: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for br in bash_runs:
        cmd = str(br.get("command", "")).strip()
        out = str(br.get("output", "")).rstrip()
        blocks.append(f"```bash\n{cmd}\n```\n\nOutput:\n```text\n{out}\n```\n")
    return "\n".join(blocks).strip()


def _build_final_prompt(components: Dict[str, Any]) -> str:
    problem = str(components.get("problem", "") or components.get("user_task_full", "")).strip()
    subagent_total = int(components.get("subagent_total") or 0)
    saved_files = str(components.get("subagent_saved_files", "")).strip() or "unknown files"
    answer_stats = str(components.get("subagent_answer_stats", "")).strip()
    bash_runs = components.get("bash_runs", []) or []
    bash_blocks = _render_bash_blocks(bash_runs) if isinstance(bash_runs, list) else ""

    stats_block = ""
    if answer_stats:
        stats_block = (
            "Answer statistics from the initial run:\n"
            "```text\n"
            + answer_stats
            + "\n```\n\n"
        )

    if not bash_blocks:
        bash_blocks = "(no bash exploration recorded)"

    return (
        "We are solving the following problem:\n\n"
        f"{problem}\n\n"
        f"I used {subagent_total} LLMs to solve it. Their responses were stored as {saved_files}.\n\n"
        f"{stats_block}"
        "I have used bash tools to explore them. Below are what I ran.\n\n"
        f"{bash_blocks}\n\n"
        "Your job is to decide the best answer. End your response with the final selection in \\boxed{...}.\n"
    )


def build_natural_prompt_components(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract natural prompt components (problem, bash commands + outputs, answer stats)."""
    saw_user_task = False
    user_task_full = ""
    last_tool_call: Dict[str, Any] | None = None

    subagent_total = 0
    subagent_saved_files = ""
    subagent_answer_stats = ""

    bash_runs: List[Dict[str, str]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            content = msg.get("content", "")
            if not saw_user_task:
                user_task_full = content
                saw_user_task = True
                continue

            normalized = _normalize_tool_result(content)
            if last_tool_call and last_tool_call.get("name") == "subagent":
                if "Responses saved to" in normalized and not subagent_saved_files:
                    # Keep just the line containing the range if possible
                    for ln in normalized.splitlines():
                        if "Responses saved to" in ln:
                            subagent_saved_files = ln.strip().split("Responses saved to", 1)[1].strip()
                            break
                stats = _extract_answer_stats_block(normalized)
                if stats and not subagent_answer_stats:
                    subagent_answer_stats = stats

            if last_tool_call and last_tool_call.get("name") == "bash":
                bash_runs.append(
                    {
                        "command": last_tool_call.get("command", ""),
                        "output": normalized,
                    }
                )

            # Clear after consuming the tool result
            last_tool_call = None

        elif role == "assistant":
            for tc in _extract_tool_calls(msg):
                name, key, value = _parse_tool_call(tc)
                if name == "commit":
                    return {
                        "user_task_full": user_task_full,
                        "problem": _extract_problem(user_task_full),
                        "subagent_total": subagent_total,
                        "subagent_saved_files": subagent_saved_files,
                        "subagent_answer_stats": subagent_answer_stats,
                        "bash_runs": bash_runs,
                    }
                if name == "subagent":
                    try:
                        subagent_total += int(value) if value else 0
                    except ValueError:
                        pass
                    last_tool_call = {"name": "subagent"}
                elif name == "bash":
                    last_tool_call = {"name": "bash", "command": value}
                else:
                    # Other tool calls aren't included in the natural prompt.
                    last_tool_call = {"name": name}

    return {
        "user_task_full": user_task_full,
        "problem": _extract_problem(user_task_full),
        "subagent_total": subagent_total,
        "subagent_saved_files": subagent_saved_files,
        "subagent_answer_stats": subagent_answer_stats,
        "bash_runs": bash_runs,
    }


def _count_subagents_from_tool_calls(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = _extract_tool_calls(msg)
        for tc in tool_calls:
            name, key, value = _parse_tool_call(tc)
            if name == "subagent" and key:
                try:
                    total += int(value)
                except ValueError:
                    continue
    return total


def main() -> None:
    args = parse_args()
    atts_dir = args.atts_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_file = atts_dir / "preds.json"
    if not preds_file.exists():
        raise FileNotFoundError(f"preds.json not found: {preds_file}")

    preds = _load_json(preds_file)
    qids = sorted(int(k) for k in preds.keys())

    if args.question_ids:
        requested = {int(x.strip()) for x in args.question_ids.split(",") if x.strip()}
        qids = [qid for qid in qids if qid in requested]

    if args.max_questions is not None:
        qids = qids[: args.max_questions]

    out_path = output_dir / "compacted_inputs.jsonl"
    with out_path.open("w") as f_out:
        for qid in qids:
            pred_info = preds[str(qid)] if str(qid) in preds else preds[qid]
            traj_file = _find_traj_file(atts_dir, qid)

            traj = _load_json(traj_file) if traj_file else {}
            messages = traj.get("messages", [])
            components = build_natural_prompt_components(messages)
            subagent_total = int(components.get("subagent_total") or 0)
            if subagent_total == 0:
                subagent_total = _count_subagents_from_tool_calls(messages)
            components["subagent_total"] = subagent_total
            final_prompt = _build_final_prompt(components)

            early_stop_metadata = traj.get("info", {}).get("early_stop_metadata", {})
            early_stopped = bool(early_stop_metadata) or pred_info.get("exit_status") == "early_stopped"

            record = {
                "question_id": qid,
                "gold": pred_info.get("gold", ""),
                "original_answer": pred_info.get("answer", ""),
                "original_correct": pred_info.get("correct", False),
                "original_exit_status": pred_info.get("exit_status", ""),
                "early_stopped": early_stopped,
                "early_stop_metadata": early_stop_metadata,
                "subagent_total": subagent_total,
                "subagent_saved_files": components.get("subagent_saved_files", ""),
                "subagent_answer_stats": components.get("subagent_answer_stats", ""),
                "problem": components.get("problem", ""),
                "user_task_full": components.get("user_task_full", ""),
                "bash_runs": components.get("bash_runs", []),
                "final_prompt": final_prompt,
                "final_prompt_char_len": len(final_prompt),
                "final_prompt_line_count": final_prompt.count("\n") + 1 if final_prompt else 0,
                "traj_file": str(traj_file) if traj_file else "",
            }
            f_out.write(json.dumps(record) + "\n")

    summary = {
        "num_questions": len(qids),
        "output_file": str(out_path),
    }
    (output_dir / "compaction_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
