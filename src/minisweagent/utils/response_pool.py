"""Response pool for managing pre-computed LLM responses."""

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from minisweagent.utils.grader import math_equal


class ResponsePool:
    """Manages pre-computed LLM responses for sampling in TTS evaluation."""

    def __init__(self, jsonl_path: str | Path | list[str | Path], seed: int = 42):
        """Initialize the response pool.

        Args:
            jsonl_path: Path(s) to JSONL file(s) with pre-computed responses.
                Can be:
                - Single path: "responses.jsonl"
                - Glob pattern: "responses_seed_*.jsonl"
                - List of paths: ["seed_0.jsonl", "seed_1.jsonl"]
                Each line should have: question_id, generation_id, vanilla_response,
                question, gold_answer, label (correctness boolean)
            seed: Random seed for reproducible sampling.
        """
        self.responses: dict[int, list[tuple[str, bool]]] = defaultdict(list)
        self.questions: dict[int, str] = {}
        self.gold_answers: dict[int, str] = {}
        self.correct_answers: dict[int, set[str]] = defaultdict(set)  # pred_answers where label=True
        self.loaded_files: list[Path] = []
        self._load(jsonl_path)
        self._seed = seed
        self.rng = random.Random(seed)

    def _resolve_paths(self, jsonl_path: str | Path | list[str | Path]) -> list[Path]:
        """Resolve input to list of file paths, expanding globs if needed."""
        # Handle list input
        if isinstance(jsonl_path, list):
            paths = [Path(p) for p in jsonl_path]
        else:
            # Handle single path/pattern
            path_str = str(jsonl_path)
            # Check if it contains glob patterns
            if any(char in path_str for char in ['*', '?', '[', ']']):
                # Expand glob pattern
                matched = glob.glob(path_str)
                if not matched:
                    raise FileNotFoundError(f"No files matched pattern: {path_str}")
                paths = [Path(p) for p in sorted(matched)]
            else:
                paths = [Path(jsonl_path)]

        # Validate all paths exist
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Response file not found: {path}")

        return paths

    def _load(self, jsonl_path: str | Path | list[str | Path]):
        """Load and group responses by question_id from one or more files."""
        paths = self._resolve_paths(jsonl_path)
        self.loaded_files = paths

        for path in paths:
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    qid = row["question_id"]

                    # Store the response text and its correctness label
                    label = row.get("label", False)
                    self.responses[qid].append((row["vanilla_response"], label))

                    # Store correct pred_answers for final answer verification
                    if label and "pred_answer" in row:
                        self.correct_answers[qid].add(str(row["pred_answer"]))

                    # Store question and gold answer (same for all responses of a question)
                    if qid not in self.questions:
                        self.questions[qid] = row["question"]
                        self.gold_answers[qid] = str(row["gold_answer"])

        # Convert to regular dict for consistent ordering
        self.responses = dict(self.responses)
        self.correct_answers = dict(self.correct_answers)

    @property
    def seed(self) -> int:
        """The random seed used for sampling."""
        return self._seed

    def sample(
        self, question_id: int, n: int, rng: random.Random | None = None
    ) -> list[tuple[str, bool]]:
        """Sample n responses without replacement.

        Args:
            question_id: The question ID to sample responses for.
            n: Number of responses to sample.
            rng: Optional RNG to use instead of the shared one.
                 Use a per-question RNG for deterministic sampling
                 regardless of thread scheduling.

        Returns:
            List of (response_text, label) tuples where label indicates correctness.
        """
        if question_id not in self.responses:
            raise KeyError(f"Question ID {question_id} not found in response pool")

        pool = self.responses[question_id]
        n = min(n, len(pool))
        _rng = rng if rng is not None else self.rng
        return _rng.sample(pool, n)

    def get_question(self, question_id: int) -> str:
        """Get the question text for a given ID."""
        if question_id not in self.questions:
            raise KeyError(f"Question ID {question_id} not found")
        return self.questions[question_id]

    def get_gold_answer(self, question_id: int) -> str:
        """Get the gold answer for evaluation."""
        if question_id not in self.gold_answers:
            raise KeyError(f"Question ID {question_id} not found")
        return self.gold_answers[question_id]

    def get_question_ids(self) -> list[int]:
        """Get all available question IDs, sorted."""
        return sorted(self.questions.keys())

    def get_num_responses(self, question_id: int) -> int:
        """Get the number of available responses for a question."""
        return len(self.responses.get(question_id, []))

    def has_correct_response(self, question_id: int) -> bool:
        """Check if any response for this question is correct."""
        if question_id not in self.responses:
            return False
        return any(label for _, label in self.responses[question_id])

    def is_correct_answer(self, question_id: int, answer: str) -> bool:
        """Check if the given answer is correct for this question.

        Uses the same math_equal verification as eval.py:
        1. First checks exact match against known correct pred_answers (fast path)
        2. Falls back to sympy-based math_equal with gold_answer (handles equivalence)
        """
        answer_str = str(answer)

        # Fast path: check if answer exactly matches a known correct pred_answer
        if question_id in self.correct_answers:
            if answer_str in self.correct_answers[question_id]:
                return True

        # Fall back to math_equal with gold answer (sympy-based equivalence check)
        if question_id in self.gold_answers:
            gold = self.gold_answers[question_id]
            try:
                return math_equal(answer_str, gold, timeout=True)
            except Exception:
                # If math_equal fails, fall back to string comparison
                return answer_str.strip() == gold.strip()

        return False

    def __len__(self) -> int:
        """Return total number of questions."""
        return len(self.questions)

    def __repr__(self) -> str:
        n_questions = len(self.questions)
        n_responses = sum(len(r) for r in self.responses.values())
        n_files = len(self.loaded_files)
        return f"ResponsePool({n_questions} questions, {n_responses} total responses from {n_files} file(s))"
