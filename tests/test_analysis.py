import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from dqa import core, retrieval
from dqa.api import app


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "knowledge").mkdir()
    monkeypatch.setattr(core, "ROOT", tmp_path)
    rng = np.random.default_rng(7)
    base = pd.DataFrame({"id": range(100), "amount": rng.normal(100, 5, 100), "region": ["West"] * 100})
    current = base.copy()
    current.loc[0, "amount"] = 10000
    current.loc[1:10, "region"] = None
    base.to_csv(tmp_path / "data" / "base.csv", index=False)
    current.to_csv(tmp_path / "data" / "now.csv", index=False)
    (tmp_path / "knowledge" / "units.md").write_text("Amount spikes: inspect cents conversion in the checkout source.", encoding="utf-8")
    return tmp_path


def test_profile_and_missingness(workspace):
    result = core.profile("now.csv")
    assert result["rows"] == 100
    assert next(c for c in result["columns"] if c["name"] == "region")["missing"] == 10
    changes = core.compare("now.csv", "base.csv")["changes"]
    assert next(c for c in changes if c["column"] == "region")["missing_change_pp"] == 10


def test_extreme_row_detected_and_identifier_excluded(workspace):
    result = core.anomalies("now.csv", "base.csv")
    assert result["features"] == ["amount"]
    spike = next(row for row in result["rows"] if row["data_row"] == 1)
    assert spike["score"] < 0


def test_path_traversal_rejected(workspace):
    with pytest.raises(ValueError):
        core.profile("../outside.csv")


def test_absolute_outside_path_rejected(workspace):
    with pytest.raises(ValueError):
        core.profile(str(workspace / "outside.csv"))


def test_empty_data_rejected(workspace):
    (workspace / "data" / "empty.csv").write_text("amount\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        core.profile("empty.csv")


def test_nonfinite_is_json_safe(workspace):
    pd.DataFrame({"amount": [1, np.inf, -np.inf, np.nan]}).to_csv(workspace / "data" / "odd.csv", index=False)
    result = core.profile("odd.csv")
    assert result["columns"][0]["non_finite"] == 2
    json.dumps(result, allow_nan=False)


def test_no_numeric_features(workspace):
    pd.DataFrame({"label": ["a"] * 25}).to_csv(workspace / "data" / "text.csv", index=False)
    with pytest.raises(ValueError, match="No usable"):
        core.anomalies("text.csv", "text.csv")


def test_small_baseline_rejected(workspace):
    pd.DataFrame({"amount": [1, 2]}).to_csv(workspace / "data" / "small.csv", index=False)
    with pytest.raises(ValueError, match="20"):
        core.anomalies("now.csv", "small.csv")


def test_schema_changes(workspace):
    pd.DataFrame({"new_field": [1, 2]}).to_csv(workspace / "data" / "new.csv", index=False)
    result = core.compare("new.csv", "base.csv")
    assert result["added_columns"] == ["new_field"]
    assert "amount" in result["removed_columns"]


def test_retrieval_citations_and_no_match(workspace):
    assert retrieval.search("cents amount conversion")[0]["source"] == "units.md"
    assert retrieval.search("zzzxxyyqq") == []


def test_missing_model_fallback(workspace, monkeypatch):
    monkeypatch.delenv("DQA_OLLAMA_MODEL", raising=False)
    result = retrieval.investigate("now.csv", "base.csv", "amount spikes", True)
    assert result["mode"] == "evidence_only"
    assert "generation_error" in result


def test_generated_rag_uses_retrieved_context(workspace, monkeypatch):
    monkeypatch.setenv("DQA_OLLAMA_MODEL", "test-model")
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "Check currency units [units.md]."}}
    def post(url, **kwargs):
        assert url.startswith("http://127.0.0.1")
        assert "units.md" in kwargs["json"]["messages"][1]["content"]
        return Response()
    monkeypatch.setattr(retrieval.httpx, "post", post)
    result = retrieval.investigate("now.csv", "base.csv", "amount spikes", True)
    assert result["mode"] == "local_rag"
    assert "[units.md]" in result["answer"]


def test_api_end_to_end(workspace):
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/datasets").json() == ["base.csv", "now.csv"]
    response = client.post("/api/investigate", json={"current": "now.csv", "baseline": "base.csv", "question": "amount spikes"})
    assert response.status_code == 200
    assert response.json()["sources"]
    assert client.get("/api/profile", params={"dataset": "../outside.csv"}).status_code == 400
    assert client.post("/api/investigate", json={"current": "now.csv", "baseline": "base.csv", "question": ""}).status_code == 422
