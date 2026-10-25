import pytest

from actrouter_client import (
    DEFAULT_URL,
    ActRouterError,
    Advisor,
    HttpRouter,
    connect,
    get_host,
)
from actrouter_client.client import task_as_function
from actrouter_client.hosts import candidate_models

MODELS = [
    {"id": "claude-haiku-4-5", "family": "anthropic", "supports_tools": True},
    {"id": "gpt-5-mini", "family": "openai", "supports_tools": True},
    {"id": "deepseek-reasoner", "family": "deepseek", "supports_tools": False},
]


# --- connect ---------------------------------------------------------------


def test_connect_defaults_to_the_hosted_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACTROUTER_URL")
    r = connect()
    assert isinstance(r, HttpRouter)
    assert r.where == DEFAULT_URL
    assert "x-api-key" not in r.headers


def test_connect_rejects_the_retired_local_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_URL", "local")
    with pytest.raises(ActRouterError, match="ACTROUTER_API_KEY"):
        connect()


def test_connect_url_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_URL", "https://router.example/")
    monkeypatch.setenv("ACTROUTER_API_KEY", "k")
    r = connect()
    assert isinstance(r, HttpRouter)
    assert r.where == "https://router.example"
    assert r.headers["x-api-key"] == "k"


# --- hosts -----------------------------------------------------------------


def test_claude_code_spells_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    host = get_host("claude-code")
    assert host.spell("claude-haiku-4-5") == "haiku"
    assert host.spell("claude-opus-4-7") == "opus"
    assert host.spell("gpt-5-mini") is None
    monkeypatch.setenv("ACTROUTER_MODEL_STYLE", "id")
    assert host.spell("claude-haiku-4-5") == "claude-haiku-4-5"


def test_host_aliases_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_host("claude").name == "claude-code"
    monkeypatch.setenv("ACTROUTER_HOST", "codex")
    assert get_host().name == "codex"
    with pytest.raises(ValueError):
        get_host("vim")


def test_candidates_follow_host_families(monkeypatch: pytest.MonkeyPatch) -> None:
    assert candidate_models(get_host("claude-code"), MODELS) == ["claude-haiku-4-5"]
    assert candidate_models(get_host("codex"), MODELS) == ["gpt-5-mini"]
    assert candidate_models(get_host("generic"), MODELS) is None
    monkeypatch.setenv("ACTROUTER_FAMILIES", "openai,deepseek")
    assert candidate_models(get_host("generic"), MODELS) == ["gpt-5-mini", "deepseek-reasoner"]
    monkeypatch.setenv("ACTROUTER_CANDIDATES", "a, b")
    assert candidate_models(get_host("codex"), MODELS) == ["a", "b"]


# --- routing (recorded API answers, see api_replay.py) ----------------------


def test_routes_tasks_with_host_spelling(claude: Advisor, api_url: str) -> None:
    search = claude.for_task("Search the repo for usages of parse_args")
    debug = claude.for_task("Debug the flaky integration test")
    assert search["host_model"] == "haiku"
    assert search["task_type"] == "code_search"
    assert debug["host_model"] in {"sonnet", "opus"}
    # Any Anthropic model: the catalog follows new releases, so no fixed list.
    assert all(a["model"].startswith("claude-") for a in debug["alternatives"])
    assert debug["routed_by"] == api_url


def test_codex_only_gets_openai_models(router: HttpRouter) -> None:
    advice = Advisor(router, host="codex").for_task("Implement a new REST endpoint", top_k=5)
    assert advice["model"].startswith("gpt-")
    assert all(a["model"].startswith("gpt-") for a in advice["alternatives"])


def test_rejected_requests_are_actrouter_errors(router: HttpRouter) -> None:
    # the API refuses anything but metadata: here, the material to work on
    with pytest.raises(ActRouterError) as e:
        router.route_task({"description": "x", "input": "code"})
    assert e.value.status == 422


def test_listings_have_the_api_shape(router: HttpRouter) -> None:
    models = router.list_models()
    assert {"id", "family", "size_tier", "strengths"} <= set(models[0])
    types = router.list_task_types()
    assert "code_search" in {t["id"] for t in types["data"]}


def test_unrunnable_choice_carries_a_note(router: HttpRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_CANDIDATES", "gpt-5-mini")
    advice = Advisor(router, host="claude-code").for_task("Summarize the thread")
    assert advice["host_model"] is None
    assert "cannot run on claude-code" in advice["note"]


# --- http ------------------------------------------------------------------


def test_http_round_trip(http_stub) -> None:
    seen = []

    def handler(method, path, body):
        seen.append((method, path, body))
        if path == "/v1/models":
            return 200, {"data": MODELS}
        return 200, {
            "chosen_model": "claude-haiku-4-5",
            "router": "heuristic",
            "ranking": [{"model_id": "claude-haiku-4-5", "score": 0.9, "reasons": ["task=code_search"]}],
            "classification": {"task_type": "code_search", "confidence": 0.98, "source": "rules"},
        }

    advice = Advisor(HttpRouter(http_stub(handler)), host="claude-code").for_task(
        "Find the config loader", preference="cost"
    )
    assert advice["host_model"] == "haiku"
    assert advice["reasons"] == ["task=code_search"]
    method, path, body = seen[-1]
    assert (method, path) == ("POST", "/v1/route/task")
    assert body["candidate_models"] == ["claude-haiku-4-5"]
    assert body["preference"] == "cost"
    assert body["task"] == {"description": "Find the config loader"}


def test_http_falls_back_to_route_on_old_deployments(http_stub) -> None:
    seen = []

    def handler(method, path, body):
        seen.append((path, body))
        if path == "/v1/route/task":
            return 404, {"detail": "Not Found"}
        if path == "/v1/models":
            return 200, {"data": MODELS}
        return 200, {"chosen_model": "gpt-5-mini", "router": "heuristic", "ranking": []}

    r = HttpRouter(http_stub(handler))
    resp = r.route_task(
        {"description": "Run the agent loop", "title": "Agent loop!", "task_kind": "tool_use", "needs": ["tools"]}
    )
    assert "predates" in resp["compat"]
    path, body = seen[-1]
    assert path == "/v1/route"
    assert body["function"] == {
        "name": "agent_loop",
        "docstring": "Run the agent loop",
        "tags": ["task:tool_use"],
    }
    assert body["candidate_models"] == ["claude-haiku-4-5", "gpt-5-mini"]


def test_http_errors(http_stub) -> None:
    url = http_stub(lambda m, p, b: (422, {"detail": "bad"}))
    with pytest.raises(ActRouterError) as e:
        HttpRouter(url).route_task({"description": "x"})
    assert e.value.status == 422
    with pytest.raises(ActRouterError, match="could not reach"):
        HttpRouter("http://127.0.0.1:9", timeout=1).list_models()


def test_task_as_function_matches_the_server() -> None:
    # what the server's TaskInfo.to_function_info() gives for this task
    task = {"description": "Fix the failing auth test", "title": "Fix auth test", "cost_budget_usd": 0.1}
    fn = task_as_function(task)
    assert fn["name"] == "fix_auth_test"
    assert fn["docstring"] == "Fix the failing auth test"
    assert fn["cost_budget_usd"] == 0.1


def test_unexpanded_placeholders_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_URL", "${user_config.router_url}")
    monkeypatch.setenv("ACTROUTER_HOST", "${HOST}")
    monkeypatch.setenv("ACTROUTER_API_KEY", "${user_config.api_key}")
    r = connect()
    assert r.where == DEFAULT_URL
    assert "x-api-key" not in r.headers
    assert get_host().name == "generic"


def test_missing_api_key_says_how_to_set_one(http_stub) -> None:
    url = http_stub(lambda m, p, b: (401, {"detail": "missing API key"}))
    with pytest.raises(ActRouterError, match="set ACTROUTER_API_KEY") as e:
        HttpRouter(url).list_models()
    assert e.value.status == 401
    with pytest.raises(ActRouterError, match="ACTROUTER_API_KEY was rejected"):
        HttpRouter(url, headers={"x-api-key": "wrong"}).list_models()
