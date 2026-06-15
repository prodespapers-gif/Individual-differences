"""
main.py
=======

Pipeline orchestrator for the manuscript "Individual Differences in
Doctoral Learning Adaptation and Well-Being: Academic Pressure, Supervisor
Support, Career Uncertainty, and the Moderating Role of Generative AI among
Chinese PhD Students."

This is the single entry point that runs the seven analysis stages in
dependency order, verifies that each stage's inputs are present before it
runs, captures per-stage status and timing, and writes a machine- and
human-readable execution report. It is the script a replicator runs to
reproduce the entire analysis from the prepared dataset onward.

The seven stages
----------------
  1. 01_data_preparation.py       -- load, screen, score; write the dataset
  2. 02_psychometric_validation.py-- reliability, CFA, invariance, formative
  3. 03_latent_profile_analysis.py-- latent profiles (person-centered)
  4. 04_predictive_modeling.py    -- AI moderation (population-average)
  5. 05_causal_heterogeneity.py   -- causal forest (individual effects)
  6. 06_qualitative_nlp.py        -- categorical/qualitative integration
  7. 07_mixed_methods_integration.py -- joint display and meta-inferences

Dependency structure
--------------------
Stage 1 produces the analysis dataset that every later stage reads. Stages
2, 3, and 4 depend only on stage 1 and are mutually independent. Stage 5
(causal forest) depends on stage 1. Stage 6 depends on stages 1 and 3 (it
crosses the categorical signal with the latent profiles). Stage 7 depends
on stages 3, 5, and 6 (it integrates profiles, treatment effects, and the
categorical signal). The orchestrator encodes these dependencies explicitly
and refuses to start a stage whose declared inputs are absent, so a failed
or skipped upstream stage produces a clear, early error rather than an
obscure downstream crash.

Why subprocess execution
------------------------
Each stage is executed as an isolated Python subprocess (python
NN_stage.py) rather than imported. This is deliberate: the stage modules
use numeric filename prefixes (not importable via a normal import
statement), each stage already has its own logging and a main() that
returns a process exit code, and process isolation guarantees that a
stage's memory, global state, and any native-library threads do not leak
into the next stage. The orchestrator captures each subprocess's exit code,
stdout/stderr tail, and wall-clock duration.

Usage
-----
  python main.py                  # run the full pipeline
  python main.py --dry-run        # show the plan and check inputs; run nothing
  python main.py --stages 1 3 5   # run a subset (dependencies still checked)
  python main.py --from-stage 3   # run stage 3 onward
  python main.py --continue-on-error   # keep going after a stage fails
  python main.py --list           # list the stages and exit

Single-sample design
--------------------
The study analyzes one sample, so the orchestrator runs one linear pipeline.
The two-sample branching (parallel per-sample stage execution and a
cross-sample alignment stage) of an earlier draft is removed; there is no
second sample to process.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from configs import CONFIG, ensure_output_directories


# ===========================================================================
# Stage specification
# ===========================================================================
@dataclass(frozen=True)
class Stage:
    """A single pipeline stage and its declared dependencies.

    Attributes
    ----------
    number:
        The stage's ordinal position (1-7).
    script:
        The script filename, executed as a subprocess from the src directory.
    name:
        A short human-readable name for logs and the report.
    depends_on:
        Stage numbers that must have run successfully before this stage.
    required_inputs:
        Files that must exist before the stage starts (resolved at run time
        against CONFIG paths). Missing inputs abort the stage with a clear
        message.
    produces:
        Files the stage is expected to create, used to verify success beyond
        the exit code and to inform downstream input checks.
    """

    number: int
    script: str
    name: str
    depends_on: tuple[int, ...] = ()
    required_inputs: tuple[Path, ...] = ()
    produces: tuple[Path, ...] = ()


def build_stage_plan() -> list[Stage]:
    """Construct the ordered stage plan with dependencies and file contracts.

    File paths are resolved from CONFIG so the plan tracks any path changes
    in one place. Only the most load-bearing inputs/outputs are declared;
    the check is intended to catch a missing upstream artifact early, not to
    enumerate every table.
    """
    tables = CONFIG.paths.tables_dir
    models = CONFIG.paths.models_dir
    data_csv = CONFIG.paths.chinese_phd_dataset
    raw_xlsx = CONFIG.paths.raw_nature_xlsx

    return [
        Stage(
            number=1,
            script="01_data_preparation.py",
            name="Data preparation",
            depends_on=(),
            required_inputs=(raw_xlsx,),
            produces=(data_csv, CONFIG.paths.codebook),
        ),
        Stage(
            number=2,
            script="02_psychometric_validation.py",
            name="Psychometric validation",
            depends_on=(1,),
            required_inputs=(data_csv,),
            produces=(tables / "table_s2_reliability.csv",),
        ),
        Stage(
            number=3,
            script="03_latent_profile_analysis.py",
            name="Latent profile analysis",
            depends_on=(1,),
            required_inputs=(data_csv,),
            produces=(
                models / "profile_assignments.csv",
                models / "profile_solution_meta.json",
            ),
        ),
        Stage(
            number=4,
            script="04_predictive_modeling.py",
            name="Predictive modeling (AI moderation)",
            depends_on=(1,),
            required_inputs=(data_csv,),
            produces=(tables / "table_s14_classical_moderation.csv",),
        ),
        Stage(
            number=5,
            script="05_causal_heterogeneity.py",
            name="Causal heterogeneity",
            depends_on=(1,),
            required_inputs=(data_csv,),
            produces=(
                models / "cate_distribution.csv",
                models / "causal_solution_meta.json",
            ),
        ),
        Stage(
            number=6,
            script="06_qualitative_nlp.py",
            name="Qualitative / categorical integration",
            depends_on=(1, 3),
            required_inputs=(data_csv, models / "profile_assignments.csv"),
            produces=(tables / "table_s29_text_availability.csv",),
        ),
        Stage(
            number=7,
            script="07_mixed_methods_integration.py",
            name="Mixed-methods integration",
            depends_on=(3, 5, 6),
            required_inputs=(
                models / "profile_assignments.csv",
                models / "cate_distribution.csv",
            ),
            produces=(
                tables / "table_s36_joint_display.csv",
                models / "integration_summary.json",
            ),
        ),
    ]


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    ensure_output_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"00_orchestrator_{timestamp}.log"

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Orchestrator logging initialized; log file at %s", log_path)
    return logger


# ===========================================================================
# Stage selection and dependency resolution
# ===========================================================================
def resolve_selected_stages(
    plan: list[Stage],
    explicit: list[int] | None,
    from_stage: int | None,
    logger: logging.Logger,
) -> list[Stage]:
    """Resolve which stages to run from the command-line selection.

    With no selection, all stages run. ``explicit`` restricts to the named
    stage numbers; ``from_stage`` runs that stage onward. The returned list
    is always in ascending stage order. Dependency satisfaction is checked
    separately at run time (a selected stage whose dependency was neither
    selected nor previously completed will be reported).
    """
    if explicit:
        selected = [s for s in plan if s.number in set(explicit)]
        logger.info("Explicit stage selection: %s", sorted(explicit))
    elif from_stage is not None:
        selected = [s for s in plan if s.number >= from_stage]
        logger.info("Running from stage %d onward", from_stage)
    else:
        selected = list(plan)
        logger.info("Running the full pipeline (all %d stages)", len(plan))
    return sorted(selected, key=lambda s: s.number)


def check_inputs_present(stage: Stage) -> list[Path]:
    """Return the list of declared required inputs that are missing."""
    return [p for p in stage.required_inputs if not p.exists()]


def check_outputs_present(stage: Stage) -> list[Path]:
    """Return the list of declared expected outputs that are missing."""
    return [p for p in stage.produces if not p.exists()]


def dependencies_satisfied(
    stage: Stage, completed: set[int],
) -> list[int]:
    """Return the stage's dependencies that are not yet completed."""
    return [d for d in stage.depends_on if d not in completed]


# ===========================================================================
# Stage execution
# ===========================================================================
def run_stage(
    stage: Stage, src_dir: Path, logger: logging.Logger,
) -> dict[str, object]:
    """Execute one stage as an isolated subprocess and capture the result.

    Returns a result record with the exit code, wall-clock duration, a tail
    of captured stdout/stderr, and whether the declared outputs appeared.
    The subprocess inherits the environment but runs from the src directory
    so that ``from configs import CONFIG`` resolves.
    """
    script_path = src_dir / stage.script
    if not script_path.exists():
        logger.error("Stage %d script not found: %s", stage.number, script_path)
        return {
            "stage": stage.number, "name": stage.name, "status": "missing_script",
            "exit_code": None, "duration_seconds": 0.0, "output_tail": "",
            "missing_outputs": [str(p) for p in stage.produces],
        }

    logger.info("-" * 72)
    logger.info("STARTING stage %d: %s (%s)", stage.number, stage.name, stage.script)
    start = time.time()

    try:
        proc = subprocess.run(
            [sys.executable, stage.script],
            cwd=str(src_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.time() - start
        exit_code = proc.returncode
        # Keep a short tail of each stream for the report.
        out_tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        err_tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        combined_tail = out_tail + (("\n[stderr]\n" + err_tail) if err_tail else "")
    except Exception as exc:  # subprocess failed to launch
        duration = time.time() - start
        logger.exception("Stage %d failed to launch: %s", stage.number, exc)
        return {
            "stage": stage.number, "name": stage.name, "status": "launch_error",
            "exit_code": None, "duration_seconds": round(duration, 2),
            "output_tail": str(exc), "missing_outputs": [],
        }

    missing_outputs = check_outputs_present(stage)
    if exit_code == 0 and not missing_outputs:
        status = "success"
        logger.info(
            "COMPLETED stage %d in %.1fs (exit 0, all declared outputs present)",
            stage.number, duration,
        )
    elif exit_code == 0 and missing_outputs:
        status = "success_missing_outputs"
        logger.warning(
            "Stage %d exited 0 but %d declared output(s) are missing: %s",
            stage.number, len(missing_outputs),
            [p.name for p in missing_outputs],
        )
    else:
        status = "failed"
        logger.error(
            "Stage %d FAILED (exit %s) after %.1fs", stage.number, exit_code, duration,
        )
        if combined_tail:
            logger.error("Stage %d output tail:\n%s", stage.number, combined_tail)

    return {
        "stage": stage.number,
        "name": stage.name,
        "script": stage.script,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 2),
        "output_tail": combined_tail,
        "missing_outputs": [str(p) for p in missing_outputs],
    }


# ===========================================================================
# Reporting
# ===========================================================================
def print_plan(
    selected: list[Stage], logger: logging.Logger,
) -> None:
    """Log the resolved execution plan, including input/output checks."""
    logger.info("=" * 72)
    logger.info("EXECUTION PLAN (%d stage(s))", len(selected))
    logger.info("=" * 72)
    for stage in selected:
        missing_inputs = check_inputs_present(stage)
        dep_str = (
            ", ".join(str(d) for d in stage.depends_on) if stage.depends_on else "none"
        )
        logger.info(
            "Stage %d: %s [%s]", stage.number, stage.name, stage.script,
        )
        logger.info("    depends on: %s", dep_str)
        if missing_inputs:
            logger.warning(
                "    MISSING inputs: %s", [p.name for p in missing_inputs],
            )
        else:
            logger.info("    inputs: present")


def write_execution_report(
    results: list[dict[str, object]],
    total_duration: float,
    logger: logging.Logger,
) -> Path:
    """Write the execution report (JSON) and log a summary table."""
    report = {
        "pipeline_version": CONFIG.pipeline_version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "root_seed": CONFIG.reproducibility.root_seed,
        "total_duration_seconds": round(total_duration, 2),
        "n_stages_run": len(results),
        "n_succeeded": sum(
            1 for r in results if r["status"].startswith("success")
        ),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "stages": results,
    }
    report_path = CONFIG.paths.logs_dir / "execution_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("=" * 72)
    logger.info("EXECUTION REPORT")
    logger.info("=" * 72)
    for r in results:
        flag = "OK " if str(r["status"]).startswith("success") else "ERR"
        logger.info(
            "  [%s] Stage %d %-38s %6.1fs  (%s)",
            flag, r["stage"], r["name"], r["duration_seconds"], r["status"],
        )
    logger.info(
        "Total: %d/%d stages succeeded in %.1fs",
        report["n_succeeded"], report["n_stages_run"], total_duration,
    )
    logger.info("Report written to %s", report_path)
    return report_path


# ===========================================================================
# Command-line interface
# ===========================================================================
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for orchestration control."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the doctoral-adaptation analysis pipeline (7 stages) in "
            "dependency order with input checking and an execution report."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the plan and check inputs without running any stage.",
    )
    parser.add_argument(
        "--stages", type=int, nargs="+", metavar="N",
        help="Run only these stage numbers (dependencies still checked).",
    )
    parser.add_argument(
        "--from-stage", type=int, metavar="N",
        help="Run from stage N onward.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue running subsequent stages after a stage fails.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the stages and exit.",
    )
    return parser.parse_args()


def main() -> int:
    """Orchestrate the pipeline according to the command-line selection."""
    args = parse_arguments()
    logger = configure_logging()
    plan = build_stage_plan()
    src_dir = Path(__file__).resolve().parent

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)

    if args.list:
        logger.info("Pipeline stages:")
        for stage in plan:
            dep = ", ".join(str(d) for d in stage.depends_on) or "none"
            logger.info(
                "  Stage %d: %s [%s] (depends on: %s)",
                stage.number, stage.name, stage.script, dep,
            )
        return 0

    selected = resolve_selected_stages(
        plan, args.stages, args.from_stage, logger,
    )
    print_plan(selected, logger)

    if args.dry_run:
        logger.info("=" * 72)
        logger.info("DRY RUN: no stages executed.")
        logger.info("=" * 72)
        # Report any missing inputs across the selection as the actionable result.
        any_missing = False
        for stage in selected:
            missing = check_inputs_present(stage)
            if missing:
                any_missing = True
                logger.warning(
                    "Stage %d would block on missing inputs: %s",
                    stage.number, [p.name for p in missing],
                )
        if not any_missing:
            logger.info("All selected stages have their declared inputs present.")
        return 0

    # Execute the selected stages in order.
    results: list[dict[str, object]] = []
    completed: set[int] = set()
    pipeline_start = time.time()

    # Pre-seed "completed" with stages whose outputs already exist and that
    # are not themselves selected, so a subset run (e.g. --from-stage 5) does
    # not falsely report unmet dependencies for already-produced upstream
    # artifacts.
    selected_numbers = {s.number for s in selected}
    for stage in plan:
        if stage.number not in selected_numbers and not check_outputs_present(stage):
            completed.add(stage.number)

    for stage in selected:
        unmet = dependencies_satisfied(stage, completed)
        if unmet:
            logger.error(
                "Stage %d (%s) cannot run: unmet dependencies %s. "
                "Run those stages first or include them in the selection.",
                stage.number, stage.name, unmet,
            )
            results.append({
                "stage": stage.number, "name": stage.name,
                "status": "skipped_unmet_dependency",
                "exit_code": None, "duration_seconds": 0.0,
                "output_tail": f"unmet dependencies: {unmet}",
                "missing_outputs": [],
            })
            if args.continue_on_error:
                continue
            logger.error("Halting (use --continue-on-error to proceed).")
            break

        missing_inputs = check_inputs_present(stage)
        if missing_inputs:
            logger.error(
                "Stage %d (%s) cannot run: missing inputs %s.",
                stage.number, stage.name, [p.name for p in missing_inputs],
            )
            results.append({
                "stage": stage.number, "name": stage.name,
                "status": "skipped_missing_input",
                "exit_code": None, "duration_seconds": 0.0,
                "output_tail": f"missing inputs: {[str(p) for p in missing_inputs]}",
                "missing_outputs": [],
            })
            if args.continue_on_error:
                continue
            logger.error("Halting (use --continue-on-error to proceed).")
            break

        result = run_stage(stage, src_dir, logger)
        results.append(result)
        if str(result["status"]).startswith("success"):
            completed.add(stage.number)
        elif not args.continue_on_error:
            logger.error(
                "Stage %d failed; halting pipeline "
                "(use --continue-on-error to proceed).", stage.number,
            )
            break

    total_duration = time.time() - pipeline_start
    write_execution_report(results, total_duration, logger)

    n_failed = sum(
        1 for r in results
        if not str(r["status"]).startswith("success")
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
