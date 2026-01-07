#!/usr/bin/env python3

"""Run TTS (Test-Time Scaling) evaluation on math problems in batch mode."""

import concurrent.futures
import json
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from rich.live import Live

from minisweagent.agents.tts import TTSAgent, TTSAgentV11
from minisweagent.config import get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.response_pool import ResponsePool

_HELP_TEXT = """Run TTS (Test-Time Scaling) orchestrator on math problems.

[not dim]
This runs an orchestrator agent that coordinates pre-computed sub-agent responses
to solve problems through intelligent consensus.

Example:
    mini-extra tts-batch responses.jsonl output/ --workers 4
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


def create_progress_tracking_agent_class(base_class):
    """Factory to create ProgressTrackingTTSAgent with a specific base class."""

    class ProgressTrackingTTSAgent(base_class):
        """TTS Agent wrapper that provides progress updates."""

        def __init__(
            self,
            *args,
            progress_manager: RunBatchProgressManager,
            instance_id: str = "",
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.progress_manager = progress_manager
            self.instance_id = instance_id

        def step(self) -> dict:
            """Override step to provide progress updates."""
            self.progress_manager.update_instance_status(
                self.instance_id,
                f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})",
            )
            return super().step()

    return ProgressTrackingTTSAgent


def update_preds_file(
    output_path: Path,
    question_id: int,
    answer: str,
    gold: str,
    exit_status: str,
):
    """Update the output JSON file with results from a single question."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())

        output_data[str(question_id)] = {
            "question_id": question_id,
            "answer": answer,
            "gold": gold,
            "correct": str(answer).strip() == str(gold).strip(),
            "exit_status": exit_status,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, question_id: int):
    """Remove a question from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        key = str(question_id)
        if key in output_data:
            del output_data[key]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_question(
    question_id: int,
    response_pool: ResponsePool,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    agent_class,
) -> None:
    """Process a single question with TTS."""
    instance_id = f"question_{question_id}"
    instance_dir = output_dir / instance_id

    # Avoid inconsistent state
    remove_from_preds_file(output_dir / "preds.json", question_id)
    (instance_dir / f"{question_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    env = LocalEnvironment(**config.get("environment", {}))

    question = response_pool.get_question(question_id)
    gold = response_pool.get_gold_answer(question_id)

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Initializing agent")

    agent = None
    extra_info = None
    exit_status = None
    result = None

    try:
        agent = agent_class(
            model,
            env,
            response_pool=response_pool,
            question_id=question_id,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(question)
    except Exception as e:
        logger.error(f"Error processing question {question_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        instance_dir.mkdir(parents=True, exist_ok=True)
        save_traj(
            agent,
            instance_dir / f"{question_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            question_id=question_id,
            gold_answer=gold,
            print_fct=logger.info,
        )
        update_preds_file(
            output_dir / "preds.json",
            question_id,
            result or "",
            gold,
            exit_status or "Unknown",
        )
        progress_manager.on_instance_end(instance_id, exit_status)
        if agent:
            agent.cleanup()


def parse_question_ids(spec: str, available_ids: list[int]) -> list[int]:
    """Parse question ID specification.

    Supports:
    - Comma-separated: "0,1,2,5"
    - Slice notation: "0:10" or "0:10:2"
    - Single ID: "5"
    """
    spec = spec.strip()
    if not spec:
        return available_ids

    # Try comma-separated
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",")]

    # Try slice notation
    if ":" in spec:
        parts = spec.split(":")
        values = [int(x) if x else None for x in parts]
        return available_ids[slice(*values)]

    # Single ID
    return [int(spec)]


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    response_jsonl: Path = typer.Argument(..., help="Path to JSONL file with pre-computed responses"),
    output_dir: Path = typer.Argument(..., help="Output directory for trajectories and predictions"),
    config_spec: Path = typer.Option("tts.yaml", "-c", "--config", help="Path to config file"),
    question_ids: str = typer.Option("", "-q", "--questions", help="Question IDs to process (e.g., '0,1,2' or '0:10')"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers"),
    seed: int = typer.Option(42, "-s", "--seed", help="Random seed for response sampling"),
    model: str | None = typer.Option(None, "-m", "--model", help="Override model name"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing questions"),
) -> None:
    # fmt: on
    """Run TTS evaluation on math problems."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    add_file_handler(output_dir / "tts_batch.log")

    # Load response pool
    logger.info(f"Loading response pool from {response_jsonl}...")
    response_pool = ResponsePool(response_jsonl, seed=seed)
    logger.info(f"Loaded {response_pool}")

    # Parse question IDs
    available_ids = response_pool.get_question_ids()
    qids = parse_question_ids(question_ids, available_ids)

    # Skip existing if not redoing
    if not redo_existing and (output_dir / "preds.json").exists():
        existing = json.loads((output_dir / "preds.json").read_text())
        existing_ids = {int(k) for k in existing.keys()}
        before = len(qids)
        qids = [q for q in qids if q not in existing_ids]
        if len(qids) != before:
            logger.info(f"Skipping {before - len(qids)} existing questions")

    if not qids:
        logger.info("No questions to process")
        return

    logger.info(f"Running on {len(qids)} questions...")

    # Load config
    config_path = get_config_path(config_spec)
    logger.info(f"Loading config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    if model is not None:
        config.setdefault("model", {})["model_name"] = model

    # Determine agent class based on config filename
    config_name = config_path.stem  # e.g., "tts" or "tts-v11"
    if "v11" in config_name:
        logger.info("Using TTSAgentV11 (stats auto-return enabled)")
        agent_class = create_progress_tracking_agent_class(TTSAgentV11)
    else:
        logger.info("Using TTSAgent (classic mode)")
        agent_class = create_progress_tracking_agent_class(TTSAgent)

    # Setup progress manager
    progress_manager = RunBatchProgressManager(
        len(qids),
        output_dir / f"exit_statuses_{time.time()}.yaml",
    )

    def process_futures(futures: dict[concurrent.futures.Future, int]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                qid = futures[future]
                logger.error(f"Error in future for question {qid}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(f"question_{qid}", e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_question,
                    qid,
                    response_pool,
                    output_dir,
                    config,
                    progress_manager,
                    agent_class,
                ): qid
                for qid in qids
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    # Print final summary
    preds_path = output_dir / "preds.json"
    if preds_path.exists():
        preds = json.loads(preds_path.read_text())
        correct = sum(1 for p in preds.values() if p.get("correct", False))
        total = len(preds)
        logger.info(f"Final accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    app()
