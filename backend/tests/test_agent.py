import json
from types import SimpleNamespace

import pytest

import app.answer.base as base
from app.agent.agent import CosmologyAgent
from app.agent.cosmology import flat_lcdm
from app.agent.tools import TOOLS, Toolbox
from app.index.store import Hit, Record

TEXT = Record("text", "2609.00001v1", "Paper A", "Results", "We find H0 = 73.2 km/s/Mpc.")
TEXT_2 = Record("text", "2609.00002v1", "Paper B", "Intro", "BAO constrain the expansion history.")
FIG = Record("figure", "2609.00001v1", "Paper A", "Figure", "Posterior on H0.", "fig1", "fig:h0", ["figures/fig1_1.png"])
PAPERS = [
    {"arxiv_id": "2609.00001v1", "title": "Paper A: the distance ladder", "authors": ["Ada Lovelace", "B. Smith"],
     "published": "2026-09-01T00:00:00Z", "abstract": "Abs A.", "figures": 1, "source_type": "latex"},
    {"arxiv_id": "2609.00002v1", "title": "Paper B: BAO", "authors": ["C. Jones"],
     "published": "2026-09-02T00:00:00Z", "abstract": "Abs B.", "figures": 0, "source_type": "latex"},
]


class FakeIndex:
    def __init__(self, text_hits, figure_hits=()):
        self.text_hits, self.figure_hits = list(text_hits), list(figure_hits)

    def search(self, query, k_text, k_figures):
        return [Hit(r, 0.8) for r in self.text_hits[:k_text]], [Hit(r, 0.7) for r in self.figure_hits[:k_figures]]


# --- lcdm_calculator ---

def test_flat_lcdm_matches_reference_values():
    # Reference: astropy FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=0).
    r = flat_lcdm(1.0, 70, 0.3)
    assert r["age_today_gyr"] == pytest.approx(13.467, abs=0.002)
    assert r["luminosity_distance_mpc"] == pytest.approx(6607.7, abs=1)
    assert r["hubble_rate_km_s_mpc"] == pytest.approx(70 * (0.3 * 8 + 0.7) ** 0.5)


def test_flat_lcdm_is_self_consistent():
    r = flat_lcdm(2.5, 67.4, 0.315)
    assert r["lookback_time_gyr"] + r["age_at_z_gyr"] == pytest.approx(r["age_today_gyr"])
    assert r["luminosity_distance_mpc"] == pytest.approx(3.5**2 * r["angular_diameter_distance_mpc"])
    assert flat_lcdm(0, 67.4, 0.315)["comoving_distance_mpc"] == 0


@pytest.mark.parametrize("z, h0, om0", [(-1, 70, 0.3), (1, 0, 0.3), (1, 70, 1.0), (1, 70, 0)])
def test_flat_lcdm_rejects_unphysical_input(z, h0, om0):
    with pytest.raises(ValueError):
        flat_lcdm(z, h0, om0)


# --- Toolbox ---

def test_tool_schemas_are_strict_compatible():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] and schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])


def test_search_labels_stay_stable_across_searches():
    box = Toolbox(FakeIndex([TEXT, TEXT_2], [FIG]), PAPERS)
    first, is_error = box.run("search_papers", {"query": "H0", "k_text": 1, "k_figures": 1})
    assert not is_error and "[S1]" in first and "[F1]" in first and "view_figure" in first
    second, _ = box.run("search_papers", {"query": "H0 again", "k_text": 2, "k_figures": 0})
    assert "[S1]" in second and "already shown earlier" in second and "[S2]" in second
    assert [s.source_id for s in box.sources] == ["S1", "F1", "S2"]


def test_search_clamps_k_and_rejects_empty_query():
    box = Toolbox(FakeIndex([TEXT] * 20), PAPERS)
    box.run("search_papers", {"query": "x", "k_text": 99, "k_figures": -3})
    assert len(box.sources) == 1  # identical records collapse to one source
    assert box.run("search_papers", {"query": "  ", "k_text": 3, "k_figures": 0})[1] is True


def test_view_figure_sends_image(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "PAPERS_DIR", tmp_path)
    (tmp_path / "2609.00001v1" / "figures").mkdir(parents=True)
    (tmp_path / "2609.00001v1" / "figures" / "fig1_1.png").write_bytes(b"\x89PNG fake")
    box = Toolbox(FakeIndex([TEXT], [FIG]), PAPERS)
    box.run("search_papers", {"query": "H0", "k_text": 1, "k_figures": 1})
    content, is_error = box.run("view_figure", {"figure_id": "[f1]"})
    assert not is_error
    assert [b["type"] for b in content] == ["text", "image"]
    assert content[1]["source"]["media_type"] == "image/png"


@pytest.mark.parametrize("figure_id", ["F1", "S1", "F9"])
def test_view_figure_only_accepts_retrieved_figures(figure_id):
    box = Toolbox(FakeIndex([TEXT]), PAPERS)
    box.run("search_papers", {"query": "H0", "k_text": 1, "k_figures": 0})
    assert box.run("view_figure", {"figure_id": figure_id})[1] is True


def test_find_papers_matches_title_and_author():
    box = Toolbox(FakeIndex([]), PAPERS)
    assert "2609.00001v1" in box.run("find_papers", {"text": "lovelace"})[0]
    assert "2609.00002v1" in box.run("find_papers", {"text": "BAO"})[0]
    assert "No paper" in box.run("find_papers", {"text": "nobody"})[0]


def test_get_paper_reads_outline_and_only_known_ids(tmp_path):
    paper_dir = tmp_path / "2609.00001v1"
    paper_dir.mkdir()
    sections = [{"path": p, "text": ""} for p in (["Intro"], ["Results"], ["Results", "H0"])]
    (paper_dir / "paper.json").write_text(json.dumps({"sections": sections}), encoding="utf-8")
    box = Toolbox(FakeIndex([]), PAPERS, papers_dir=tmp_path)
    content, is_error = box.run("get_paper", {"arxiv_id": "arXiv:2609.00001v1"})
    assert not is_error and "Abs A." in content and "Intro\nResults\n  H0" in content
    for bad in ["../../backend/.env", "2609.99999v1"]:
        assert box.run("get_paper", {"arxiv_id": bad})[1] is True


def test_bad_tool_calls_become_error_results():
    box = Toolbox(FakeIndex([]), PAPERS)
    assert box.run("delete_everything", {}) == ("Unknown tool 'delete_everything'.", True)
    assert box.run("lcdm_calculator", {"z": 1, "H0": 70})[1] is True  # missing argument
    assert box.run("lcdm_calculator", {"z": 1, "H0": 70, "Om0": 2})[1] is True


# --- agent loop ---

def _usage():
    return SimpleNamespace(input_tokens=100, output_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=50)


def _tool_use(name, tool_input, id_="tu_1"):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=tool_input)


def _response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason, model="claude-test", usage=_usage())


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def test_agent_runs_tools_and_checks_citations():
    client = FakeClient([
        _response([
            SimpleNamespace(type="thinking", thinking=""),
            _tool_use("search_papers", {"query": "H0", "k_text": 2, "k_figures": 0}, "tu_1"),
            _tool_use("lcdm_calculator", {"z": 0, "H0": 73.2, "Om0": 0.3}, "tu_2"),
        ], "tool_use"),
        _response([SimpleNamespace(type="text", text="H0 = 73.2 [S1], age 12.9 Gyr [S9].")], "end_turn"),
    ])
    steps = []
    run = CosmologyAgent(client=client).run("What is H0?", FakeIndex([TEXT, TEXT_2]), PAPERS, on_step=steps.append)

    assert [s.tool for s in run.steps] == ["search_papers", "lcdm_calculator"] and steps == run.steps
    assert run.steps[1].output.startswith("Flat ΛCDM with H0 = 73.2")
    assert [c.source_id for c in run.answer.citations] == ["S1"]
    assert run.answer.unknown_markers == ["S9"]
    assert run.answer.input_tokens == 300 and run.answer.output_tokens == 20
    assert run.answer.provider == "agent" and run.answer.stop_reason == "end_turn"

    # Both results go back in one user message, after the unchanged assistant turn.
    assistant, user = client.requests[1]["messages"][1:]
    assert assistant["content"][0].type == "thinking"
    assert [r["tool_use_id"] for r in user["content"]] == ["tu_1", "tu_2"]
    assert client.requests[0]["tool_choice"] == {"type": "auto"}


def test_agent_switches_tools_off_on_last_turn():
    search = _tool_use("search_papers", {"query": "H0", "k_text": 1, "k_figures": 0})
    client = FakeClient([
        _response([search], "tool_use"),
        _response([SimpleNamespace(type="text", text="Answer [S1].")], "end_turn"),
    ])
    run = CosmologyAgent(client=client, max_turns=2).run("q", FakeIndex([TEXT]), PAPERS)
    assert client.requests[1]["tool_choice"] == {"type": "none"}
    assert client.requests[1]["messages"][-1]["content"][-1]["type"] == "text"  # the wrap-up note
    assert run.answer.text == "Answer [S1]."


def test_agent_reports_refusal():
    client = FakeClient([_response([], "refusal")])
    run = CosmologyAgent(client=client).run("q", FakeIndex([]), PAPERS)
    assert run.answer.stop_reason == "refusal" and "declined" in run.answer.text
