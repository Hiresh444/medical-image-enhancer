"""mdimg pipeline package — reusable medical imaging QA pipeline.

Import the unified entry point:

    from pipeline.runner import run_pipeline
"""


def __getattr__(name: str):
    if name == "run_pipeline":
        from pipeline.runner import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_pipeline"]
