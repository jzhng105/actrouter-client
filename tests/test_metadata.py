import textwrap
from pathlib import Path

import pytest

from actrouter_client import Advisor, FunctionNotFound, function_metadata_from_source

SOURCE = textwrap.dedent(
    '''
    from actdecor import summarize_with_llm

    SECRET = "sk-never-sent"

    @summarize_with_llm(model="openai/gpt-5-mini")
    def summarize_article(text: str, *, bullets: int = 3) -> str:
        """Summarize a news article in 3 bullet points."""
        return SECRET

    class Triage:
        def classify(self, ticket: str) -> "Literal['bug', 'feature']":
            """Classify a support ticket."""

    def classify(x):
        pass

    class A:
        def run(self): ...

    class B:
        def run(self): ...
    '''
)


@pytest.fixture()
def module(tmp_path: Path) -> Path:
    path = tmp_path / "src" / "news" / "pipeline.py"
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE)
    return path


def test_extracts_static_metadata_only(module: Path, tmp_path: Path) -> None:
    meta = function_metadata_from_source(module, "summarize_article", root=tmp_path)
    assert meta["name"] == "summarize_article"
    assert meta["module"] == "news.pipeline"
    assert meta["signature"] == "(text: str, *, bullets: int=3) -> str"
    assert meta["parameter_types"] == ["str", "int"]
    assert meta["return_type"] == "str"
    assert meta["decorators"] == ["summarize_with_llm(model='openai/gpt-5-mini')"]
    assert meta["docstring"] == "Summarize a news article in 3 bullet points."
    assert "sk-never-sent" not in repr(meta)


def test_methods_and_ambiguity(module: Path) -> None:
    meta = function_metadata_from_source(module, "Triage.classify")
    assert meta["qualname"] == "Triage.classify"
    assert function_metadata_from_source(module, "classify")["qualname"] == "classify"
    assert function_metadata_from_source(module, "A.run")["qualname"] == "A.run"
    with pytest.raises(FunctionNotFound, match="ambiguous"):
        function_metadata_from_source(module, "run")
    with pytest.raises(FunctionNotFound, match="no function"):
        function_metadata_from_source(module, "nope")


def test_hints_are_added(module: Path) -> None:
    meta = function_metadata_from_source(module, "summarize_article", task_kind="tldr", cost_budget_usd=None)
    assert meta["task_kind"] == "tldr"
    assert "cost_budget_usd" not in meta


def test_route_from_source(module: Path, claude: Advisor) -> None:
    advice = claude.for_source(str(module), "Triage.classify")
    assert advice["task_type"] == "classification"
    assert advice["function"]["qualname"] == "Triage.classify"
    assert advice["host_model"] in {"haiku", "sonnet", "opus"}
