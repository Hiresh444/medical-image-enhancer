import argparse
import logging
import os
import sys

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

logger = logging.getLogger("mdimg")

# Default model — env var takes precedence, CLI flag overrides both
_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Medical Imaging Quality Assurance (DICOM QA)"
    )
    parser.add_argument("--input", required=True, help="Path to a single DICOM file")
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for report and visuals (default: outputs)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display matplotlib window (still saves figures)",
    )

    # --- GenAI flags ---
    parser.add_argument(
        "--genai",
        action="store_true",
        help="Enable GenAI agentic mode (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"OpenAI model for GenAI agents (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=4,
        help="Max tuning iterations for GenAI TuningAgent (default: 4)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="GenAI generates plan JSON but does not execute enhancement",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable metadata redaction (default: redact enabled)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- Logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    # --- Run pipeline via the unified runner ---
    from pipeline.runner import run_pipeline

    try:
        context = run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            genai=args.genai,
            model=args.model,
            max_iters=args.max_iters,
            plan_only=args.plan_only,
            save_artifacts=True,
            no_show=args.no_show,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        logger.exception("Pipeline failed")
        return 1

    # --- Plan-only mode ---
    if context.get("plan_only") and context.get("plan"):
        print("\n=== GenAI Enhancement Plan (JSON) ===\n")
        print(context["plan"].model_dump_json(indent=2))
        if context.get("stop_reason"):
            print(f"\nStop reason: {context['stop_reason']}")
        return 0

    # --- Print report ---
    report_md = context.get("report_md", "")
    if report_md:
        print(report_md)

    # --- Fallback warning ---
    if context.get("genai_fell_back"):
        print(f"\nWARNING: GenAI failed ({context.get('genai_error')}). Used deterministic fallback.")

    # --- Show matplotlib window ---
    if not args.no_show:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
