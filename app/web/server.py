"""API web de l'Assistant Data AI + service de l'interface HTML (index.html à la racine).

Lancement : python -m app.server   (http://localhost:7860)
"""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
import uuid
from collections import OrderedDict
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.llm.openai_client import OpenAIClient
from app.llm.web_research import web_research
from app.pages.guide import build_guide_html
from app.pages.research import RESEARCH_EXAMPLES, sources_html
from app.pipeline.feedback import save_feedback
from app.web import analysis as A

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = ROOT / "index.html"
STATIC_DIR = ROOT / "static"
FAVICON = STATIC_DIR / "img" / "robot.svg"

app = FastAPI(title="Assistant Data AI", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# index.html ouvert directement (origine « null ») ou servi par un autre serveur local
# (ex. Live Server de VS Code) : seules les origines locales sont autorisées.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(null|https?://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------
# Fichiers à télécharger (exports) : jeton -> chemin
# ---------------------------------------------------------

_DOWNLOADS: OrderedDict[str, Path] = OrderedDict()
_DOWNLOADS_LOCK = threading.Lock()
MAX_DOWNLOADS = 300


def register_download(path: Any) -> str | None:
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    token = uuid.uuid4().hex
    with _DOWNLOADS_LOCK:
        _DOWNLOADS[token] = file
        while len(_DOWNLOADS) > MAX_DOWNLOADS:
            _DOWNLOADS.popitem(last=False)
    return f"/api/download/{token}"


# ---------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convertit récursivement les types pandas/numpy/Decimal en JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):          # numpy scalaire
        return to_jsonable(value.item())
    return str(value)


def _value(item: Any) -> Any:
    if isinstance(item, dict) and item.get("__update__"):
        return item["value"] if item.get("visible", True) else None
    return item


def figure_json(fig: Any) -> dict | None:
    return json.loads(fig.to_json()) if fig is not None else None


def panel_payload(panel: dict[str, Any]) -> dict[str, Any]:
    """Panneau de résultats (éléments de analysis.build_reply) -> JSON pour l'interface."""
    downloads = []
    for key, label, kind in (("download", "CSV", "csv"), ("excel", "Excel", "excel"), ("pptx", "PowerPoint", "pptx")):
        url = register_download(_value(panel.get(key)))
        if url:
            downloads.append({"label": label, "kind": kind, "url": url})

    return {
        "badges": _value(panel.get("kpi")) or "",
        "highlights": _value(panel.get("highlights")) or "",
        "table": to_jsonable(_value(panel.get("table"))),
        "chart": figure_json(_value(panel.get("chart"))),
        "downloads": downloads,
        "sql": _value(panel.get("sql")) or "",
        "plain": _value(panel.get("plain")) or "",
        "trace": _value(panel.get("trace")) or "",
        "warnings": _value(panel.get("warnings")) or "",
        "glossary": _value(panel.get("glossary")) or A.EMPTY_GLOSSARY,
        "suggestions": panel.get("suggestions") or [],
        "alerts": _value(panel.get("alerts")) if panel.get("alerts") is not None else None,
    }


def ndjson(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        for event in events:
            yield json.dumps(to_jsonable(event), ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


def run_with_progress(work: Callable[[Callable[[Any], None]], Any], poll: float = 0.25) -> Iterator[tuple[str, Any]]:
    """Exécute ``work(report)`` dans un thread ; produit ("progress", x) puis ("done"|"error", y)."""
    updates: queue.Queue = queue.Queue()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = work(updates.put)
        except Exception as exc:  # noqa: BLE001 - remonté à l'appelant
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    while worker.is_alive() or not updates.empty():
        try:
            yield "progress", updates.get(timeout=poll)
        except queue.Empty:
            continue
    worker.join()
    if "error" in box:
        yield "error", box["error"]
    else:
        yield "done", box.get("result")


# ---------------------------------------------------------
# Modèles de requête
# ---------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Message] = Field(default_factory=list)
    memory: list[dict[str, Any]] = Field(default_factory=list)
    year: str = "Toutes"
    region: str = "Toutes"
    focus_region: str | None = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str = ""
    liked: bool
    sql: str | None = None


class MissionRequest(BaseModel):
    objective: str
    year: str = "Toutes"
    region: str = "Toutes"


class ResearchRequest(BaseModel):
    question: str


def _filters(year: str, region: str) -> dict[str, str]:
    return {k: v for k, v in (("Année", year), ("Région", region)) if v and v != "Toutes"}


# ---------------------------------------------------------
# Pages et fichiers
# ---------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(FAVICON, media_type="image/svg+xml")


@app.get("/api/download/{token}", include_in_schema=False)
def download(token: str):
    with _DOWNLOADS_LOCK:
        path = _DOWNLOADS.get(token)
    if path is None or not path.is_file():
        raise HTTPException(404, "Fichier expiré : relancez l'analyse.")
    return FileResponse(path, filename=path.name)


# ---------------------------------------------------------
# API
# ---------------------------------------------------------


@app.get("/api/config")
def config():
    return {
        "welcome": A.WELCOME,
        "examples": A.EXAMPLES,
        "mission_examples": A.MISSION_EXAMPLES,
        "research_examples": RESEARCH_EXAMPLES,
        "empty_glossary": A.EMPTY_GLOSSARY,
    }


@app.get("/api/filters")
def filters():
    try:
        options = A.get_pipeline().get_agent().filter_options()
    except Exception:
        logger.exception("Chargement des filtres impossible")
        options = {}
    return {"years": [str(v) for v in options.get("annee", [])], "regions": [str(v) for v in options.get("region", [])]}


@app.get("/api/alerts")
def alerts():
    try:
        triggered = A.get_pipeline().get_agent().check_alerts()
    except Exception:
        logger.exception("Vérification des alertes impossible")
        triggered = []
    return {"html": _value(A.alerts_update(triggered)) or ""}


@app.get("/api/match-region")
def match_region(name: str):
    return {"region": A._match_region(name)}


@app.post("/api/ask")
def ask(req: AskRequest):
    """Question en langage naturel : flux NDJSON (étapes, puis résultat)."""
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Saisissez une question.")

    filters = _filters(req.year, req.region)
    if req.focus_region:
        filters.pop("Région", None)
    history = [m.model_dump() for m in req.history if m.content and m.content != A.WELCOME]

    def work(report):
        return A.get_pipeline().run_chat(
            question, history, req.memory, filters, req.focus_region, on_step=report
        )

    def events():
        start = time.perf_counter()
        for kind, payload in run_with_progress(work):
            if kind == "progress":
                yield {"type": "step", "label": str(payload)}
            elif kind == "error":
                logger.error("Erreur pendant l'exécution du pipeline", exc_info=payload)
                yield {
                    "type": "result",
                    "message": "Je rencontre actuellement un problème pour récupérer ces informations. "
                               "Vous pouvez réessayer dans un instant.",
                    "panel": panel_payload(A.empty_panel()),
                    "memo": None,
                }
            else:
                try:
                    message, panel = A.build_reply(payload, time.perf_counter() - start)
                    memo = (getattr(payload, "metadata", None) or {}).get("memo")
                    yield {
                        "type": "result",
                        "message": message,
                        "panel": panel_payload(panel),
                        "memo": memo if memo and memo.get("sql") else None,
                    }
                except Exception:
                    logger.exception("Mise en forme de la réponse impossible")
                    yield {
                        "type": "result",
                        "message": "Je n'ai pas pu mettre en forme cette réponse. Réessayez dans un instant.",
                        "panel": panel_payload(A.empty_panel()),
                        "memo": None,
                    }

    return ndjson(events())


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    try:
        save_feedback(req.question, req.answer, req.liked, req.sql)
    except Exception:
        logger.exception("Enregistrement de l'avis impossible")
        raise HTTPException(500, "Avis non enregistré.")
    return {
        "message": "Merci ! Cette réponse servira d'exemple à l'assistant."
        if req.liked else "Merci, votre avis est enregistré. N'hésitez pas à reformuler la question."
    }


@app.post("/api/mission")
def mission(req: MissionRequest):
    """Agent analyste : flux NDJSON de la progression, puis rapport complet."""
    objective = req.objective.strip()
    if not objective:
        raise HTTPException(400, "Décrivez la mission à confier à l'agent.")
    filters = _filters(req.year, req.region)

    def work(report):
        return A.get_pipeline().run_mission(objective, filters, on_update=report)

    def events():
        start = time.perf_counter()
        yield {"type": "progress", "html": A.mission_progress_html({"phase": "plan", "steps": []})}
        for kind, payload in run_with_progress(work, poll=0.3):
            if kind == "progress":
                yield {"type": "progress", "html": A.mission_progress_html(payload)}
            elif kind == "error":
                logger.error("Mission impossible", exc_info=payload)
                yield {
                    "type": "result",
                    "progress": A.mission_progress_html({"phase": "error", "steps": []}),
                    "report": "Je n'ai pas pu mener cette mission. Réessayez dans un instant.",
                    "steps": [], "downloads": [],
                }
            else:
                yield {"type": "result", **mission_payload(payload, time.perf_counter() - start)}

    return ndjson(events())


def mission_payload(mission, elapsed: float) -> dict[str, Any]:
    if mission.error and not mission.report:
        return {
            "progress": A.mission_progress_html(mission.snapshot()),
            "report": f"⚠️ {mission.error}", "steps": [], "downloads": [],
        }
    steps = [
        {"markdown": A._step_markdown(i, step), "chart": figure_json(A._step_chart(step))}
        for i, step in enumerate(mission.steps[: A.MISSION_MAX_STEPS], start=1)
    ]
    downloads = []
    for label, kind, path in (
        ("Présentation (PowerPoint)", "pptx", A.export_mission_pptx(mission)),
        ("Données (Excel)", "excel", A.export_mission_excel(mission)),
        ("Rapport (Markdown)", "md", A.export_mission_markdown(mission)),
    ):
        url = register_download(path)
        if url:
            downloads.append({"label": label, "kind": kind, "url": url})
    return {
        "progress": A.mission_progress_html(mission.snapshot(), elapsed),
        "report": mission.report,
        "steps": steps,
        "downloads": downloads,
    }


@lru_cache(maxsize=1)
def research_client() -> OpenAIClient:
    return OpenAIClient()


@app.post("/api/research")
def research(req: ResearchRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Saisissez une question de recherche.")
    start = time.perf_counter()
    try:
        result = web_research(research_client(), question)
    except Exception as error:
        logger.exception("Recherche web impossible")
        return JSONResponse({"error": str(error)[:300]}, status_code=502)
    return {
        "answer": result.answer or "_Aucune réponse obtenue._",
        "sources": [{"title": s.title, "url": s.url, "domain": s.domain} for s in result.sources],
        "sources_html": sources_html(result.sources),
        "elapsed": round(time.perf_counter() - start, 1),
    }


@app.get("/api/guide", response_class=HTMLResponse)
def guide(q: str = ""):
    return build_guide_html(A.GLOSSARY, q)
