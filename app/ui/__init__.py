"""UI-only helpers shared across ClockWork's pages.

Everything in this package renders widgets or manages Streamlit session state.
**None of it computes anything scientific** — that all lives in ``core/``.

The split is deliberate and load-bearing for the planned headless-batch refactor:
when the analyses become runnable without Streamlit, nothing under ``app/ui/``
has to be untangled, because none of it is on the path between a parameter and a
result. Page files are then thin enough to read as
"gather params -> call core -> store result -> render".

Pages import these as top-level names (``from ui.guards import require_dataset``)
because ``app/`` is placed on ``sys.path`` by the entry point, ``app/ClockWork.py``.
"""
