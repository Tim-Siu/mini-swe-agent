"""TTS (Test-Time Scaling) Agent for orchestrating sub-agent consensus."""

import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    FormatError,
    Submitted,
)


class EarlyStopped(Exception):
    """Raised when early stopping is triggered (no correct answer in sampled pool)."""

    def __init__(self, message: str = "", metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


from minisweagent.utils.response_pool import ResponsePool


class TTSAgentConfig(AgentConfig):
    """Extended config for TTS agent."""

    max_attempts: int = 32  # Maximum sub-agents to spawn
    tool_call_format: str = "default"  # "default" or "glm"


class TTSAgent(DefaultAgent):
    """Orchestrator agent for Test-Time Scaling.

    This agent coordinates multiple pre-computed sub-agent responses to solve
    problems through intelligent consensus. It supports four action types:
    - bash: Execute shell commands in the working directory
    - subagent: Spawn N sub-agents (sample from response pool)
    - stats: Aggregate and display answer statistics
    - commit: Submit final answer
    """

    def __init__(
        self,
        model,
        env,
        response_pool: ResponsePool,
        question_id: int,
        **kwargs,
    ):
        """Initialize TTS agent.

        Args:
            model: The LLM model for orchestration decisions.
            env: The execution environment.
            response_pool: Pool of pre-computed sub-agent responses.
            question_id: ID of the question being solved.
            **kwargs: Additional config arguments.
        """
        super().__init__(model, env, config_class=TTSAgentConfig, **kwargs)
        self.response_pool = response_pool
        self.question_id = question_id
        self.working_dir = tempfile.mkdtemp(prefix="tts_")
        self.spawned_count = 0
        # Per-question deterministic RNG (independent of thread scheduling)
        self._sample_rng = random.Random(f"{response_pool.seed}_{question_id}")
        # Accumulate metadata across subagent calls
        self._subagent_call_log: list[dict] = []
        # Note: max_attempts is already available in templates via config.model_dump()

    def parse_action(self, response: dict) -> dict:
        """Extract action type and content from response.

        Dispatches to format-specific parser based on config.tool_call_format.
        """
        if self.config.tool_call_format == "glm":
            return self._parse_action_glm(response)
        else:
            return self._parse_action_default(response)

    def _parse_action_default(self, response: dict) -> dict:
        """Parse default XML-style tool calls with parameters.

        Matches:
        <tool_call>
        <function=bash|subagent|stats|commit>
        <parameter=PARAM_NAME>
        content
        </parameter>
        </function>
        </tool_call>
        """
        content = response["content"]

        # Match: <tool_call><function=TYPE>...<parameter=NAME>CONTENT</parameter>...</function></tool_call>
        # or for stats: <tool_call><function=stats></function></tool_call> (no parameter)
        pattern = r"<tool_call>\s*<function=(bash|subagent|stats|commit)>\s*(.*?)\s*</function>\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        if len(matches) == 1:
            action_type, inner_content = matches[0]

            # Extract parameter value if present
            param_pattern = r"<parameter=\w+>\s*(.*?)\s*</parameter>"
            param_matches = re.findall(param_pattern, inner_content, re.DOTALL)

            if param_matches:
                action_content = param_matches[0].strip()
            else:
                # For stats or other parameterless calls
                action_content = inner_content.strip()

            return {
                "type": action_type,
                "action": action_content,
                **response,
            }

        raise FormatError(
            self.render_template(self.config.format_error_template, actions=matches)
        )

    def _parse_action_glm(self, response: dict) -> dict:
        """Parse GLM native tool call format.

        First checks for structured tool_calls in the API response (OpenAI format),
        then falls back to parsing text content for GLM's native XML format:
        <tool_call>function_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
        """
        # First, check for structured tool_calls in the API response
        # These are at response["extra"]["response"]["choices"][0]["message"]["tool_calls"]
        try:
            tool_calls = response.get("extra", {}).get("response", {}).get("choices", [{}])[0].get("message", {}).get("tool_calls")
            if tool_calls and len(tool_calls) == 1:
                tool_call = tool_calls[0]
                # Handle both dict and object formats
                if hasattr(tool_call, "function"):
                    func = tool_call.function
                    action_type = func.name if hasattr(func, "name") else func.get("name")
                    args_str = func.arguments if hasattr(func, "arguments") else func.get("arguments", "{}")
                else:
                    func = tool_call.get("function", {})
                    action_type = func.get("name")
                    args_str = func.get("arguments", "{}")

                # Parse arguments JSON
                if isinstance(args_str, str):
                    args = json.loads(args_str) if args_str else {}
                else:
                    args = args_str or {}

                # Extract the first argument value (count, command, or answer)
                if args:
                    action_content = str(list(args.values())[0])
                else:
                    action_content = ""

                if action_type in ("bash", "subagent", "stats", "commit"):
                    return {
                        "type": action_type,
                        "action": action_content,
                        **response,
                    }
        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
            pass  # Fall through to text parsing

        # Fall back to parsing text content for GLM's native XML format
        content = response.get("content", "")

        # Match: <tool_call>FUNCTION_NAME<arg_key>...</arg_key><arg_value>...</arg_value>...</tool_call>
        pattern = r"<tool_call>(bash|subagent|stats|commit)(.*?)</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        if len(matches) == 1:
            action_type, args_content = matches[0]

            # Extract argument value (we only care about the first arg_value for our tools)
            arg_pattern = r"<arg_key>(\w+)</arg_key><arg_value>(.*?)</arg_value>"
            arg_matches = re.findall(arg_pattern, args_content, re.DOTALL)

            if arg_matches:
                # Take the first argument's value
                action_content = arg_matches[0][1].strip()
            else:
                # For parameterless calls like stats
                action_content = ""

            return {
                "type": action_type,
                "action": action_content,
                **response,
            }

        raise FormatError(
            self.render_template(self.config.format_error_template, actions=matches)
        )

    def execute_action(self, action: dict) -> dict:
        """Route to appropriate action handler."""
        action_type = action["type"]

        if action_type == "commit":
            raise Submitted(action["action"])
        elif action_type == "subagent":
            return self._execute_subagent(action)
        elif action_type == "stats":
            return self._execute_stats(action)
        elif action_type == "bash":
            return self._execute_bash(action)
        else:
            raise FormatError(f"Unknown action type: {action_type}")

    def _extract_answer(self, response: str) -> str:
        """Extract the last \\boxed{} answer from a response."""
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, response)
        return matches[-1] if matches else "NO_ANSWER"

    def _execute_subagent(self, action: dict) -> dict:
        """Sample and write N sub-agent responses to files, sorted by answer frequency."""
        try:
            n = int(action["action"])
        except ValueError:
            return {
                "output": f"Error: Invalid number of agents: {action['action']}",
                "returncode": 1,
                "action": action["action"],
            }

        n = min(n, self.config.max_attempts)
        if n <= 0:
            return {
                "output": "Error: Number of agents must be positive",
                "returncode": 1,
                "action": action["action"],
            }

        # Sample responses from pool (returns list of (response_text, label) tuples)
        # Uses per-question RNG for deterministic sampling regardless of threading
        try:
            sampled = self.response_pool.sample(
                self.question_id, n, rng=self._sample_rng
            )
        except KeyError as e:
            return {
                "output": f"Error: {e}",
                "returncode": 1,
                "action": action["action"],
            }

        # Log metadata before any sorting
        call_metadata = {
            "call_index": len(self._subagent_call_log),
            "count": len(sampled),
            "samples": [
                {
                    "response_hash": hashlib.sha256(resp.encode()).hexdigest()[:16],
                    "extracted_answer": self._extract_answer(resp),
                    "label": label,
                }
                for resp, label in sampled
            ],
        }
        self._subagent_call_log.append(call_metadata)

        # Extract answers and pair with responses and labels
        # Format: (response_text, extracted_answer, label)
        responses_with_answers = [
            (resp, self._extract_answer(resp), label) for resp, label in sampled
        ]

        # Count answer frequencies
        answer_counts = Counter(ans for _, ans, _ in responses_with_answers)

        # Sort responses by answer frequency (most common first), then by answer for stability
        responses_with_answers.sort(
            key=lambda x: (-answer_counts[x[1]], x[1])
        )

        # Write responses to files in sorted order
        # Store labels for later use (e.g., early stopping)
        self._sampled_labels = []
        for i, (resp, _, label) in enumerate(responses_with_answers):
            path = Path(self.working_dir) / f"{i}.txt"
            path.write_text(resp)
            self._sampled_labels.append(label)

        self.spawned_count = len(sampled)

        return {
            "output": f"Spawned {len(sampled)} agents. Responses saved to 0.txt - {len(sampled) - 1}.txt",
            "returncode": 0,
            "action": action["action"],
        }

    def _execute_stats(self, action: dict) -> dict:
        """Aggregate answers from response files using \\boxed{} extraction."""
        results = []
        # Match \boxed{...} - handle nested braces
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"

        txt_files = sorted(
            Path(self.working_dir).glob("*.txt"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
        )

        if not txt_files:
            return {
                "output": "No response files found. Run subagent first.",
                "returncode": 1,
                "action": action["action"],
            }

        for filepath in txt_files:
            if not filepath.stem.isdigit():
                continue
            content = filepath.read_text()
            matches = re.findall(pattern, content)
            # Take the last \boxed{} match (usually the final answer)
            answer = matches[-1] if matches else "NO_ANSWER"
            results.append((int(filepath.stem), answer))

        if not results:
            return {
                "output": "No valid response files found.",
                "returncode": 1,
                "action": action["action"],
            }

        # Group by answer
        answer_to_indices: dict[str, list[int]] = {}
        for idx, answer in results:
            if answer not in answer_to_indices:
                answer_to_indices[answer] = []
            answer_to_indices[answer].append(idx)

        # Sort by frequency (descending)
        sorted_answers = sorted(
            answer_to_indices.items(), key=lambda x: -len(x[1])
        )

        # Build output table
        lines = ["| Index   | Extracted Answer |", "| ------- | ---------------- |"]

        for answer, indices in sorted_answers:
            indices.sort()
            if len(indices) == 1:
                idx_str = str(indices[0])
            elif indices == list(range(indices[0], indices[-1] + 1)):
                # Consecutive range
                idx_str = f"{indices[0]}-{indices[-1]}"
            else:
                # Non-consecutive, show first-last
                idx_str = f"{indices[0]}-{indices[-1]}"

            # Truncate long answers for display
            display_answer = answer if len(answer) <= 16 else answer[:13] + "..."
            lines.append(f"| {idx_str:<7} | {display_answer:<16} |")

        return {
            "output": "\n".join(lines),
            "returncode": 0,
            "action": action["action"],
        }

    def _execute_bash(self, action: dict) -> dict:
        """Execute bash command in working directory."""
        output = self.env.execute(action["action"], cwd=self.working_dir)
        return output | {"action": action["action"]}

    def get_sampled_rollout_metadata(self) -> dict:
        """Get accumulated metadata for all subagent calls.

        Returns a dict with subagent_calls, total_budget_used,
        sampled_answers (frequency dict), and gold_answer.
        """
        all_answers = []
        for call in self._subagent_call_log:
            for s in call["samples"]:
                all_answers.append(s["extracted_answer"])
        return {
            "subagent_calls": self._subagent_call_log,
            "total_budget_used": sum(c["count"] for c in self._subagent_call_log),
            "sampled_answers": dict(Counter(all_answers)),
            "gold_answer": self.response_pool.get_gold_answer(self.question_id),
        }

    def cleanup(self):
        """Clean up working directory."""
        if Path(self.working_dir).exists():
            shutil.rmtree(self.working_dir)

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass


class TTSAgentV11(TTSAgent):
    """TTS Agent V11 - Automatically returns stats after spawning subagents.

    This version simplifies the workflow by automatically displaying answer
    statistics immediately after spawning parallel subagents, eliminating the
    need for a separate stats tool call.

    Supports forced budget mode where the first turn is overridden with a
    templated response that spawns a fixed number of agents.
    """

    def __init__(
        self,
        model,
        env,
        response_pool: ResponsePool,
        question_id: int,
        forced_budget: int | None = None,
        **kwargs,
    ):
        """Initialize TTS agent V11.

        Args:
            model: The LLM model for orchestration decisions.
            env: The execution environment.
            response_pool: Pool of pre-computed sub-agent responses.
            question_id: ID of the question being solved.
            forced_budget: If set, override first turn to spawn this many agents.
            **kwargs: Additional config arguments.
        """
        super().__init__(model, env, response_pool, question_id, **kwargs)
        self.forced_budget = forced_budget
        self._first_turn_injected = False

    def step(self) -> dict:
        """Execute one agent step, with optional first-turn injection."""
        # Inject forced budget on first turn if configured
        if not self._first_turn_injected and self.forced_budget is not None:
            self._first_turn_injected = True
            return self._inject_forced_first_turn()
        return super().step()

    def _inject_forced_first_turn(self) -> dict:
        """Inject a templated first turn that spawns the forced budget."""
        # Create templated response with generic THOUGHT
        # Use GLM format if configured, otherwise default format
        if self.config.tool_call_format == "glm":
            tool_call = f"<tool_call>subagent<arg_key>count</arg_key><arg_value>{self.forced_budget}</arg_value></tool_call>"
        else:
            tool_call = f"<tool_call>\n<function=subagent>\n<parameter=count>\n{self.forced_budget}\n</parameter>\n</function>\n</tool_call>"

        forced_response = {
            "content": f"THOUGHT: Spawning agents to solve this problem.\n\n{tool_call}",
            "extra": {"usage": {}},  # No actual model call
        }

        # Add as assistant message
        self.add_message("assistant", **forced_response)

        # Parse and execute the action
        return self.get_observation(forced_response)

    def _execute_subagent(self, action: dict) -> dict:
        """Sample and write N sub-agent responses, then automatically show stats."""
        # First, execute the regular subagent spawning
        result = super()._execute_subagent(action)

        # If spawning failed, return the error
        if result["returncode"] != 0:
            return result

        # Check for early stopping: if no correct answer in sampled responses
        # Use pre-computed labels from the rollout data (set by parent class)
        has_correct = any(self._sampled_labels)

        if not has_correct:
            # Use metadata already accumulated by parent's _execute_subagent
            metadata = self.get_sampled_rollout_metadata()
            metadata["early_stop_reason"] = "no_correct_answer_in_sampled_responses"
            raise EarlyStopped(
                f"No correct answer found in {self.spawned_count} sampled responses",
                metadata=metadata,
            )

        # Now automatically generate and append stats
        stats_result = self._execute_stats(action)

        # Combine the outputs
        combined_output = result["output"]
        if stats_result["returncode"] == 0:
            combined_output += "\n\nAnswer Statistics:\n" + stats_result["output"]
        else:
            # If stats somehow failed, just note it
            combined_output += "\n\n(Note: Could not generate statistics)"

        return {
            "output": combined_output,
            "returncode": result["returncode"],
            "action": result["action"],
        }

    def execute_action(self, action: dict) -> dict:
        """Route to appropriate action handler, excluding stats tool."""
        action_type = action["type"]

        if action_type == "commit":
            raise Submitted(action["action"])
        elif action_type == "subagent":
            return self._execute_subagent(action)
        elif action_type == "stats":
            # In v11, stats is not a separate tool
            raise FormatError("stats is not available in v11. Stats are automatically shown after spawning subagents.")
        elif action_type == "bash":
            return self._execute_bash(action)
        else:
            raise FormatError(f"Unknown action type: {action_type}")
