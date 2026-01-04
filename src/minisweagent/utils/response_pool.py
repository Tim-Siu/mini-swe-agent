"""Response pool for managing pre-computed LLM responses."""

import json
import random
from collections import defaultdict
from pathlib import Path


class ResponsePool:
    """Manages pre-computed LLM responses for sampling in TTS evaluation."""

    def __init__(self, jsonl_path: str | Path, seed: int = 42):
        """Initialize the response pool.

        Args:
            jsonl_path: Path to JSONL file with pre-computed responses.
                Each line should have: question_id, generation_id, vanilla_response,
                question, gold_answer
            seed: Random seed for reproducible sampling.
        """
        self.responses: dict[int, list[str]] = defaultdict(list)
        self.questions: dict[int, str] = {}
        self.gold_answers: dict[int, str] = {}
        self._load(jsonl_path)
        self.rng = random.Random(seed)

    def _load(self, jsonl_path: str | Path):
        """Load and group responses by question_id."""
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Response file not found: {jsonl_path}")

        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                qid = row["question_id"]

                # Store the response text
                self.responses[qid].append(row["vanilla_response"])

                # Store question and gold answer (same for all responses of a question)
                if qid not in self.questions:
                    self.questions[qid] = row["question"]
                    self.gold_answers[qid] = str(row["gold_answer"])

        # Convert to regular dict for consistent ordering
        self.responses = dict(self.responses)

    def sample(self, question_id: int, n: int) -> list[str]:
        """Sample n responses without replacement.

        Args:
            question_id: The question ID to sample responses for.
            n: Number of responses to sample.

        Returns:
            List of response strings (vanilla_response field).
        """
        if question_id not in self.responses:
            raise KeyError(f"Question ID {question_id} not found in response pool")

        pool = self.responses[question_id]
        n = min(n, len(pool))
        return self.rng.sample(pool, n)

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

    def __len__(self) -> int:
        """Return total number of questions."""
        return len(self.questions)

    def __repr__(self) -> str:
        n_questions = len(self.questions)
        n_responses = sum(len(r) for r in self.responses.values())
        return f"ResponsePool({n_questions} questions, {n_responses} total responses)"
