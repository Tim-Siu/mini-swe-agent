"""TTS (Test-Time Scaling) Agent for orchestrating sub-agent consensus."""

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
from minisweagent.utils.response_pool import ResponsePool


class TTSAgentConfig(AgentConfig):
    """Extended config for TTS agent."""

    max_attempts: int = 32  # Maximum sub-agents to spawn


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
        # Note: max_attempts is already available in templates via config.model_dump()

    def parse_action(self, response: dict) -> dict:
        """Extract action type and content from response.

        Matches XML-style tool calls with parameters:
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

        # Sample responses from pool
        try:
            responses = self.response_pool.sample(self.question_id, n)
        except KeyError as e:
            return {
                "output": f"Error: {e}",
                "returncode": 1,
                "action": action["action"],
            }

        # Extract answers and pair with responses
        responses_with_answers = [
            (resp, self._extract_answer(resp)) for resp in responses
        ]

        # Count answer frequencies
        answer_counts = Counter(ans for _, ans in responses_with_answers)

        # Sort responses by answer frequency (most common first), then by answer for stability
        responses_with_answers.sort(
            key=lambda x: (-answer_counts[x[1]], x[1])
        )

        # Write responses to files in sorted order
        for i, (resp, _) in enumerate(responses_with_answers):
            path = Path(self.working_dir) / f"{i}.txt"
            path.write_text(resp)

        self.spawned_count = len(responses)

        return {
            "output": f"Spawned {len(responses)} agents. Responses saved to 0.txt - {len(responses) - 1}.txt",
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
        forced_response = {
            "content": f"THOUGHT: Spawning agents to solve this problem.\n\n<tool_call>\n<function=subagent>\n<parameter=count>\n{self.forced_budget}\n</parameter>\n</function>\n</tool_call>",
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
