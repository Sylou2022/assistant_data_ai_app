import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.llm.web_research import ResearchResult, Source
from app.web import analysis as A
from app.web import server


class FakeAgent:
    def filter_options(self):
        return {"annee": [2024, 2025], "region": ["Nord", "Sud"]}

    def check_alerts(self):
        return []


class FakePipeline:
    def __init__(self):
        self.agent = FakeAgent()

    def get_agent(self):
        return self.agent

    def run_chat(self, question, history, memory, filters, focus_region, on_step=None):
        on_step("Lecture du catalogue")
        on_step("Exécution de la requête")
        return SimpleNamespace(
            status="ok",
            question=question,
            sql="SELECT Region, SUM(CA) AS CA FROM Ventes GROUP BY Region",
            warnings=[],
            analysis_text="Le **Sud** devance le Nord.",
            dataframe=pd.DataFrame({"Region": ["Nord", "Sud"], "Chiffre_Affaires": [10.0, 20.0]}),
            forecast_result=None,
            last_updated=None,
            explanation="Somme du CA par région.",
            tables_used=["Ventes"],
            columns_used=["CA"],
            assumptions=[],
            metadata={
                "memo": {"question": question, "sql": "SELECT 1"},
                "suggestions": ["CA par mois", "Top produits"],
                "plain": ["Somme du chiffre d'affaires par région."],
                "chart": {"type": "bar"},
            },
        )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(A, "get_pipeline", lambda: FakePipeline())
    return TestClient(server.app)


def test_index_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Assistant Data" in page.text and "static/js/app.js" in page.text
    assert client.get("/static/css/style.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/favicon.ico").status_code == 200


def test_cors_allows_local_pages_only(client):
    ok = client.get("/api/config", headers={"Origin": "http://127.0.0.1:5500"})
    assert ok.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"
    assert client.get("/api/config", headers={"Origin": "null"}).headers.get("access-control-allow-origin") == "null"
    other = client.get("/api/config", headers={"Origin": "https://exemple.com"})
    assert "access-control-allow-origin" not in other.headers


def test_config_and_filters(client):
    config = client.get("/api/config").json()
    assert config["examples"] and config["mission_examples"] and config["research_examples"]
    assert client.get("/api/filters").json() == {"years": ["2024", "2025"], "regions": ["Nord", "Sud"]}
    assert client.get("/api/alerts").json() == {"html": ""}


def test_ask_streams_steps_then_result(client):
    res = client.post("/api/ask", json={"question": "CA par région ?", "year": "2024"})
    assert res.status_code == 200
    events = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert [e["label"] for e in events if e["type"] == "step"] == ["Lecture du catalogue", "Exécution de la requête"]

    result = events[-1]
    assert result["type"] == "result"
    assert "Sud" in result["message"]
    assert result["memo"]["sql"] == "SELECT 1"

    panel = result["panel"]
    assert panel["sql"].startswith("SELECT")
    assert panel["table"]["rows"] and panel["table"]["numeric"] == [False, True]
    assert panel["chart"] and "data" in panel["chart"]
    assert panel["suggestions"] == ["CA par mois", "Top produits"]

    kinds = {d["kind"]: d["url"] for d in panel["downloads"]}
    assert "csv" in kinds
    csv = client.get(kinds["csv"])
    assert csv.status_code == 200 and "Nord" in csv.text


def test_ask_rejects_empty_question(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400


def test_download_unknown_token(client):
    assert client.get("/api/download/inconnu").status_code == 404


def test_research(client, monkeypatch):
    monkeypatch.setattr(
        server, "web_research",
        lambda _client, q: ResearchResult(q, "Synthèse", [Source("Insee", "https://www.insee.fr/a")]),
    )
    monkeypatch.setattr(server, "research_client", lambda: object())
    data = client.post("/api/research", json={"question": "Inflation ?"}).json()
    assert data["answer"] == "Synthèse"
    assert data["sources"][0]["domain"] == "insee.fr"
    assert "research-source" in data["sources_html"]


def test_guide_filter(client):
    html = client.get("/api/guide", params={"q": "prevision"}).text
    assert "Prévision" in html
