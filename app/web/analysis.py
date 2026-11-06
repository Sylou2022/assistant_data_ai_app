"""Logique d'analyse de l'Assistant Data AI (sans interface).

Formatage des résultats, graphiques Plotly, exports (CSV, Excel, PowerPoint),
prévisions, missions de l'agent analyste et glossaire. Utilisé par l'API web
(app/web/server.py), qui sérialise ces éléments en JSON pour l'interface HTML.
"""

from __future__ import annotations

import html
import logging
import re
import tempfile
import time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.llm.openai_client import OpenAIClient
from app.pipeline.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Constantes
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]   # dossier app/


EXAMPLES = [
    "Quel est le chiffre d'affaires par région ?",
    "Quels sont mes 5 produits les plus rentables ?",
    "Compare 2024 et 2023 : qu'est-ce qui a changé ?",
    "Analyse les ventes et identifie les tendances ou anomalies",
    "Prévois la marge pour les 6 prochains mois",
]

WELCOME = (
    "Bonjour 👋 Je suis votre assistant Data. "
    "Posez-moi simplement votre question en langage naturel "
    "et je vous aiderai à trouver l'information recherchée."
)


EMPTY_GLOSSARY = (
    '<div class="dynamic-glossary-empty">'
    "Les termes utilisés dans la réponse apparaîtront ici."
    "</div>"
)

# Couleurs
INDIGO = "#1e3a6e"   # bleu nuit (couleur principale des graphiques)
GOLD = "#c9a227"
TEXT_DARK = "#0b1a3a"

TIME_WORDS = (
    "annee", "année", "year", "mois", "month", "jour", "day",
    "trimestre", "quarter", "semaine", "week", "date",
    "periode", "période",
)
NON_ADDITIVE_WORDS = ("taux", "moyenne", "ratio", "pourcent", "%", "prix", "pct", "avg")

LABEL_REPLACEMENTS = (
    (r"\bchiffre affaires\b", "chiffre d'affaires"),
    (r"\bquantite\b", "quantité"),
    (r"\bregion\b", "région"),
    (r"\bannee\b", "année"),
    (r"\bcategorie\b", "catégorie"),
    (r"\bcout\b", "coût"),
    (r"\bbenefice\b", "bénéfice"),
    (r"\bca\b", "CA"),
)

MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)
MONTH_COLUMN_NAMES = ("mois", "month")




# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_pipeline() -> DataPipeline:
    """Instancie le pipeline une seule fois avec OpenAI."""
    return DataPipeline(llm_client=OpenAIClient())


def update(value: Any = None, visible: bool = True) -> dict[str, Any]:
    """Élément du panneau de résultats : une valeur et sa visibilité."""
    return {"__update__": True, "value": value, "visible": visible}


# ---------------------------------------------------------
# Détection des colonnes et formatage
# ---------------------------------------------------------


def _norm(name: Any) -> str:
    return str(name).lower()


def is_id_column(name: Any) -> bool:
    return _norm(name).endswith("id")


def is_time_column(name: Any) -> bool:
    return any(word in _norm(name) for word in TIME_WORDS)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes Decimal/objets numériques en vrais nombres."""
    df = df.copy().reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype != object:
            continue
        values = df[col].dropna()
        if len(values) and all(
            isinstance(v, (Decimal, int, float)) and not isinstance(v, bool)
            for v in values
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return month_names(df)


def month_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace un numéro de mois (8) par son nom (août), en catégorie ordonnée."""
    for col in df.columns:
        if _norm(col) not in MONTH_COLUMN_NAMES:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().all() and values.between(1, 12).all() and (values % 1 == 0).all():
            df[col] = pd.Categorical(
                [MONTHS_FR[int(v) - 1] for v in values],
                categories=MONTHS_FR,
                ordered=True,
            )

    return df


def measure_columns(df: pd.DataFrame) -> list:
    """Colonnes d'indicateurs chiffrés (hors identifiants et dates)."""
    return [
        col
        for col in df.select_dtypes(include="number").columns
        if not is_id_column(col) and not is_time_column(col)
    ]


def label_columns(df: pd.DataFrame) -> list:
    """Colonnes d'axes d'analyse (région, produit, mois...)."""
    measures = set(measure_columns(df))
    return [c for c in df.columns if c not in measures and not is_id_column(c)]


def pretty_column(name: Any) -> str:
    """Chiffre_Affaires_Total -> Chiffre d'affaires total."""
    text = str(name).replace("_", " ").strip().lower()
    text = re.sub(r"^nom ", "", text)
    for pattern, replacement in LABEL_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text[:1].upper() + text[1:]


def detect_unit(name: Any) -> str:
    """Devine l'unité d'un indicateur d'après son nom."""
    n = _norm(name)

    if any(word in n for word in ("taux", "pct", "pourcent", "ratio", "%")):
        return "%"

    euro_words = ("chiffre", "affaires", "marge", "montant", "prix", "cout", "revenu", "benefice")
    if any(word in n for word in euro_words) or re.search(r"(^|_)ca($|_)", n):
        return "€"

    if "quantit" in n:
        return "unités"

    return ""


def format_number(value: int) -> str:
    return f"{value:,}".replace(",", "\u00a0")


def plural(count: int, word: str) -> str:
    return f"{format_number(count)} {word}{'s' if count > 1 else ''}"


def format_value(value: Any, unit: str = "", compact: bool = False) -> str:
    """Formate un nombre à la française : 13 136 952 € ou 13,1 M€."""
    number = float(value)

    if unit == "%" and abs(number) <= 1:
        number *= 100

    if compact and unit == "€" and abs(number) >= 1e6:
        scaled, suffix = (
            (number / 1e9, "Md€") if abs(number) >= 1e9 else (number / 1e6, "M€")
        )
        return f"{scaled:.1f}".replace(".", ",") + "\u00a0" + suffix

    if unit == "%":
        decimals = 0 if number.is_integer() else 1
    else:
        decimals = 0 if number.is_integer() or abs(number) >= 1000 else 2

    text = f"{number:,.{decimals}f}".replace(",", "\u00a0").replace(".", ",")
    return f"{text}\u00a0{unit}" if unit else text


def format_label(value: Any) -> str:
    """Affiche une valeur d'axe sans décimales parasites (2024, pas 2024.0)."""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def is_additive(col: Any, values: pd.Series, unit: str) -> bool:
    """Une somme ou une part du total a-t-elle un sens pour cet indicateur ?"""
    return (
        values.min() >= 0
        and unit != "%"
        and not any(word in _norm(col) for word in NON_ADDITIVE_WORDS)
    )


# ---------------------------------------------------------
# Tableau : noms lisibles, nombres formatés, couleurs
# ---------------------------------------------------------


def readable_names(df: pd.DataFrame) -> list[str]:
    names = [pretty_column(c) for c in df.columns]
    return names if len(set(names)) == len(names) else [str(c) for c in df.columns]




# ---------------------------------------------------------
# Exports
# ---------------------------------------------------------


def table_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Tableau prêt pour l'interface : en-têtes lisibles, valeurs formatées, barres relatives."""
    data = df.reset_index(drop=True)
    measures = set(measure_columns(data))
    names = readable_names(data)
    numeric = [col in measures for col in data.columns]

    rows = []
    for _, record in data.iterrows():
        rows.append([
            "" if pd.isna(v) else (format_value(v, detect_unit(col)) if col in measures else format_label(v))
            for col, v in zip(data.columns, record)
        ])

    bars: list[float] | None = None
    first = next((c for c in data.columns if c in measures), None)
    if first is not None and len(data) > 1:
        top = data[first].max()
        if pd.notna(top) and top > 0:
            bars = [0.0 if pd.isna(v) else round(max(float(v), 0.0) / float(top), 4) for v in data[first]]

    return {
        "columns": names,
        "numeric": numeric,
        "rows": rows,
        "bars": bars,
        "bar_column": names[list(data.columns).index(first)] if bars else None,
    }


def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = readable_names(df)
    return out


def export_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.mkdtemp()) / "resultats.csv"
    export_frame(df).to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    return str(path)


def export_excel(df: pd.DataFrame) -> str | None:
    """Export Excel mis en forme. Retourne None si openpyxl est absent."""
    try:
        from openpyxl.styles import Font, PatternFill

        path = Path(tempfile.mkdtemp()) / "resultats.xlsx"
        out = export_frame(df)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Résultats")
            sheet = writer.sheets["Résultats"]

            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="4F46E5")

            for idx, col in enumerate(out.columns, start=1):
                longest = max([len(str(col))] + [len(str(v)) for v in out[col].head(200)])
                letter = sheet.cell(row=1, column=idx).column_letter
                sheet.column_dimensions[letter].width = min(longest + 3, 50)

        return str(path)

    except Exception:
        logger.warning("Export Excel indisponible (pip install openpyxl)", exc_info=True)
        return None


# ---------------------------------------------------------
# Réponse de repli (pipeline classique, sans agent)
# ---------------------------------------------------------


def analysis_label_column(question: str, df: pd.DataFrame) -> Any | None:
    """Détermine la dimension réellement analysée à partir de la question."""
    labels = label_columns(df)
    if not labels:
        return None

    q = _norm(question)

    if "mois" in q or "month" in q:
        for col in labels:
            if "nom_mois" in _norm(col):
                return col
        for col in labels:
            if _norm(col) in ("mois", "month"):
                return col

    if "région" in q or "region" in q:
        for col in labels:
            if "region" in _norm(col):
                return col

    if "produit" in q:
        for col in labels:
            if "produit" in _norm(col):
                return col

    if "année" in q or "annee" in q or "year" in q:
        for col in labels:
            n = _norm(col)
            if "annee" in n or "year" in n:
                return col

    return labels[0]


def build_answer(question: str, df: pd.DataFrame) -> str:
    """Phrase de réponse directe avec le chiffre clé."""
    measures = measure_columns(df)
    data = df.dropna(subset=measures[:1]) if measures else df
    if not measures or data.empty:
        return ""

    labels = label_columns(df)
    value_col = measures[0]
    analysis_label = analysis_label_column(question, df)
    unit = detect_unit(value_col)
    metric = pretty_column(value_col).lower()

    q = question.lower()
    wants_min = any(w in q for w in ("moins", "minimum", "faible", "pire", "dernier", "plus bas"))
    wants_max = not wants_min and any(
        w in q for w in ("plus", "meilleur", "maximum", "premier", "top", "élevé")
    )

    ranked = data.sort_values(value_col, ascending=False)
    row = ranked.iloc[-1] if wants_min else ranked.iloc[0]
    value = f"**{format_value(row[value_col], unit, compact=True)}**"

    if not labels:
        return f"{pretty_column(value_col)} : {value}."

    if analysis_label is not None:
        label = f"**{format_label(row[analysis_label])}**"

        if is_time_column(analysis_label) and len(data) > 1:
            phrase = "Point bas" if wants_min else "Pic"
            return f"{phrase} en {label} — {metric} : {value}."

        if wants_min:
            return f"{label} affiche la valeur la plus faible — {metric} : {value}."

        if wants_max or len(data) > 1:
            return f"{label} affiche la valeur la plus élevée — {metric} : {value}."

        return f"{label} — {metric} : {value}."

    return f"{metric} : {value}."


def build_insights(question: str, df: pd.DataFrame) -> list[str]:
    """Synthèse métier concise (repli sans agent)."""
    measures = measure_columns(df)
    if not measures:
        return []

    data = df.dropna(subset=[measures[0]]).copy()
    if len(data) < 2:
        return []

    value_col = measures[0]
    unit = detect_unit(value_col)
    analysis_label = analysis_label_column(question, data)
    if analysis_label is None:
        return []

    ranked = data.sort_values(value_col, ascending=False)
    best, last = ranked.iloc[0], ranked.iloc[-1]
    values = ranked[value_col]

    def fmt(value: Any) -> str:
        return format_value(value, unit, compact=True)

    dimension = pretty_column(analysis_label).lower()

    lines = [
        f"**{dimension.capitalize()} le plus élevé** : "
        f"**{format_label(best[analysis_label])}** — **{fmt(best[value_col])}**.",
        f"**{dimension.capitalize()} le plus faible** : "
        f"**{format_label(last[analysis_label])}** — **{fmt(last[value_col])}**.",
    ]

    if is_additive(value_col, values, unit):
        lines.append(f"**Total** : **{fmt(values.sum())}** · **moyenne** : **{fmt(values.mean())}**.")
    else:
        lines.append(f"**Moyenne** : **{fmt(values.mean())}**.")

    return lines


# ---------------------------------------------------------
# Panneau de droite : KPI, traçabilité, suggestions
# ---------------------------------------------------------


def format_last_updated(value: Any) -> str | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        return f"{ts.day:02d}/{ts.month:02d}/{ts.year} à {ts.hour:02d}:{ts.minute:02d}"
    except Exception:
        return str(value)


def delta_html(pct: float | None, label: str = "") -> str:
    """▲ +4,2 % vs 2023 (vert) / ▼ -3,1 % (rouge)."""
    if pct is None:
        return ""
    if abs(pct) < 0.0005:
        arrow, css = "▬", "flat"
    elif pct > 0:
        arrow, css = "▲", "up"
    else:
        arrow, css = "▼", "down"
    text = f"{pct * 100:+.1f} %".replace(".", ",")
    return (
        f'<span class="delta {css}">{arrow} {text}</span> '
        f'<span class="delta-label">{html.escape(label)}</span>'
    )


def build_kpi_html(
    df: pd.DataFrame,
    elapsed: float | None,
    question: str = "",
    last_updated: Any = None,
    compare: dict | None = None,
    data_until: str | None = None,
) -> str:
    """En-tête : pastilles d'état uniquement (les chiffres clés sont dans « À retenir »)."""
    badges = [
        f'<span class="kpi-badge">{plural(len(df), "résultat")}</span>',
        '<span class="kpi-badge green">🟢 SQL connecté</span>',
        '<span class="kpi-badge green">🔒 Requête validée · lecture seule</span>',
    ]

    if elapsed is not None:
        seconds = f"{elapsed:.1f}".replace(".", ",")
        badges.append(f'<span class="kpi-badge">⏱ {seconds} s</span>')

    if data_until:
        badges.append(
            f'<span class="kpi-badge">📅 Données jusqu\'à {html.escape(str(data_until))}</span>'
        )

    last_updated_text = format_last_updated(last_updated)
    if last_updated_text:
        badges.append(
            f'<span class="kpi-badge">↻ Mise à jour : {html.escape(last_updated_text)}</span>'
        )

    return f'<div class="kpi-section"><div class="kpi-status-row">{"".join(badges)}</div></div>'


def build_trace(result, elapsed: float | None) -> str:
    """Traçabilité : d'où vient le chiffre."""
    parts: list[str] = []

    explanation = getattr(result, "explanation", None)
    if explanation:
        parts.append(f"**Méthode** : {explanation}")

    tables = getattr(result, "tables_used", None)
    if tables:
        parts.append("**Tables utilisées** : " + ", ".join(f"`{t}`" for t in tables))

    columns = getattr(result, "columns_used", None)
    if columns:
        parts.append("**Colonnes utilisées** : " + ", ".join(f"`{c}`" for c in columns))

    assumptions = getattr(result, "assumptions", None)
    if assumptions:
        parts.append("**Hypothèses**\n" + "\n".join(f"- {a}" for a in assumptions))

    parts.append("**Contrôle de sécurité** : requêtes validées avant exécution (lecture seule).")

    if elapsed is not None:
        parts.append(f"**Temps de réponse** : {f'{elapsed:.1f}'.replace('.', ',')} s")

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Graphique générique piloté par l'assistant
# ---------------------------------------------------------


def _resolve_column(df: pd.DataFrame, name: Any):
    if not name:
        return None
    lookup = {str(c).lower(): c for c in df.columns}
    return lookup.get(str(name).strip().lower())


def _hover_format(unit: str, values: list[float]) -> str:
    """Format Plotly du survol, accolade fermante incluse."""
    small = bool(values) and max(abs(v) for v in values) < 100
    pattern = ",.2f" if small else ",.0f"
    if unit == "%":
        return ",.1f} %"
    if unit == "€":
        return f"{pattern}}} €"
    if unit == "unités":
        return f"{pattern}}} unités"
    return f"{pattern}}}"


def _is_orderable(series: pd.Series) -> bool:
    """Colonne dont l'ordre naturel est fiable (dates, nombres, catégories ordonnées)."""
    return (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_numeric_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

GEOJSON_PATH = BASE_DIR / "data" / "regions.geojson"


def _plain_key(text: Any) -> str:
    import unicodedata

    t = unicodedata.normalize("NFD", str(text).lower().replace("’", "'"))
    return "".join(c for c in t if unicodedata.category(c) != "Mn").strip()


@lru_cache(maxsize=1)
def _load_regions_geojson():
    """Charge le GeoJSON des régions (None si le fichier est absent)."""
    if not GEOJSON_PATH.is_file():
        return None, {}
    import json

    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    names = {
        _plain_key(f["properties"]["nom"]): f["properties"]["nom"]
        for f in gj.get("features", [])
        if "nom" in (f.get("properties") or {})
    }
    return gj, names


def _region_bbox(gj, feature_name: str):
    """(lon_min, lon_max, lat_min, lat_max) d'une région du GeoJSON."""
    for f in gj.get("features", []):
        if (f.get("properties") or {}).get("nom") != feature_name:
            continue
        pts: list = []

        def walk(c):
            if c and isinstance(c[0], (int, float)):
                pts.append(c)
            else:
                for s in c:
                    walk(s)

        walk(f["geometry"]["coordinates"])
        lons, lats = [p[0] for p in pts], [p[1] for p in pts]
        pad = max(max(lons) - min(lons), max(lats) - min(lats)) * 0.25
        return min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad
    return None


def _france_map(data, x_col, value_col, title, hover_fmt, pretty_label, selected=None):
    gj, names = _load_regions_geojson()
    if not gj:
        return None

    rows = [
        (names.get(_plain_key(r)), float(v))
        for r, v in zip(data[x_col], data[value_col])
    ]
    rows = [(n, v) for n, v in rows if n]
    if len(rows) < max(3, int(len(data) * 0.6)):
        return None   # trop de noms non reconnus : on garde les barres

    hover = (
        f"<b>%{{location}}</b><br>{pretty_label} : "
        f"<b>%{{z:{hover_fmt}</b><extra></extra>"
    )

    fig = go.Figure(
        go.Choropleth(
            geojson=gj,
            locations=[n for n, _ in rows],
            z=[v for _, v in rows],
            featureidkey="properties.nom",
            colorscale=[[0, "#E6ECF7"], [1, INDIGO]],
            marker_line_color="white",
            marker_line_width=0.8,
            colorbar=dict(title=None, tickformat="~s"),
            hovertemplate=hover,
        )
    )

    sel_name = names.get(_plain_key(selected)) if selected else None
    bbox = _region_bbox(gj, sel_name) if sel_name else None

    if bbox:
        value = next(v for n, v in rows if n == sel_name) if any(n == sel_name for n, _ in rows) else None
        if value is not None:
            fig.add_trace(
                go.Choropleth(
                    geojson=gj,
                    locations=[sel_name],
                    z=[value],
                    featureidkey="properties.nom",
                    colorscale=[[0, "#F59E0B"], [1, "#F59E0B"]],
                    showscale=False,
                    marker_line_color="#111827",
                    marker_line_width=2.5,
                    hovertemplate=hover,
                )
            )
        fig.update_geos(
            visible=False,
            projection_type="mercator",
            lonaxis_range=[bbox[0], bbox[1]],
            lataxis_range=[bbox[2], bbox[3]],
        )
    else:
        fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color=TEXT_DARK)),
        margin=dict(l=0, r=0, t=70, b=0),
        height=340,
        paper_bgcolor="white",
        geo=dict(bgcolor="white"),
    )
    return fig


def _focus_map(rdf: pd.DataFrame, region: str, years: list[str]):
    """Carte de toutes les régions, recadrée sur la région choisie."""
    labels, measures = label_columns(rdf), measure_columns(rdf)
    x = next((c for c in labels if "region" in _norm(c)), None)
    if x is None or not measures or not rdf[x].is_unique:
        return None
    y = measures[0]
    hover = _hover_format(detect_unit(y), [float(v) for v in rdf[y]])
    title = f"{pretty_column(y)} par région{_period_suffix(years)} — {region}"
    return _france_map(rdf, x, y, title, hover, pretty_column(y), selected=region)


def _rank_chip(rdf: pd.DataFrame, region: str):
    """Rang et part de la région dans le total (calculés en code)."""
    labels, measures = label_columns(rdf), measure_columns(rdf)
    x = next((c for c in labels if "region" in _norm(c)), None)
    if x is None or not measures:
        return None
    y = measures[0]
    unit = detect_unit(y)
    if not is_additive(y, rdf[y].dropna(), unit):
        return None
    ranked = rdf.dropna(subset=[y]).sort_values(y, ascending=False).reset_index(drop=True)
    hit = ranked.index[ranked[x].map(_plain_key) == _plain_key(region)]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    share = float(ranked.loc[i, y]) / float(ranked[y].sum()) * 100
    return (
        "",
        f"Rang {i + 1} / {len(ranked)}",
        f"{region} · {format_value(ranked.loc[i, y], unit, compact=True)} · "
        f"{share:.1f} % du total".replace(".", ","),
    )


def _extreme_colors(values: list[float], base: str = INDIGO):
    """Maximum en vert, minimum en rouge, le reste en couleur de base."""
    if len(values) < 3 or max(values) == min(values):
        return base
    hi, lo = max(values), min(values)
    return ["#16A34A" if v == hi else "#EF4444" if v == lo else base for v in values]


def _ordered_points(df: pd.DataFrame, value_col) -> list[tuple[str, float]] | None:
    """Série chronologique [(libellé, valeur)] si le résultat a un axe temps fiable."""
    labels = label_columns(df)
    year_col = next((c for c in labels if any(w in _norm(c) for w in ("annee", "year"))), None)
    month_col = next((c for c in labels if _norm(c) in MONTH_COLUMN_NAMES), None)
    date_col = next(
        (c for c in labels if any(w in _norm(c) for w in ("date", "jour", "periode", "période"))),
        None,
    )
    x = month_col or date_col or year_col
    if x is None:
        return None

    data = df.dropna(subset=[value_col])
    if month_col and x == month_col and year_col and data[year_col].nunique() > 1:
        data = data.sort_values([year_col, month_col])
        names = [f"{format_label(m)} {format_label(y)}" for m, y in zip(data[month_col], data[year_col])]
    elif _is_orderable(data[x]):
        data = data.sort_values(x)
        names = [format_label(v) for v in data[x]]
    else:
        return None
    return list(zip(names, data[value_col].astype(float)))


def make_chart_from_spec(
    df: pd.DataFrame,
    spec: dict | None = None,
    anomalies: list | None = None,
):
    """Graphique Plotly piloté par l'assistant (type, x, y, years), avec repli automatique."""
    spec = spec or {}
    kind = str(spec.get("type") or "auto").lower()
    requested = kind

    if kind == "none" or len(df) < 2:
        return None

    measures, labels = measure_columns(df), label_columns(df)

    y_cols = [
        c
        for c in (_resolve_column(df, p) for p in str(spec.get("y") or "").split(","))
        if c is not None and c in measures
    ] or measures[:1]

    if not y_cols or not labels:
        return None

    year_col = next((c for c in labels if any(w in _norm(c) for w in ("annee", "year"))), None)
    month_col = next((c for c in labels if _norm(c) in MONTH_COLUMN_NAMES), None)
    date_col = next(
        (c for c in labels if any(w in _norm(c) for w in ("date", "jour", "periode", "période"))),
        None,
    )

    x_col = _resolve_column(df, spec.get("x")) or month_col or date_col or year_col or labels[0]
    if x_col in y_cols:
        return None

    data = df.dropna(subset=y_cols[:1]).copy()
    time_like = is_time_column(x_col) or pd.api.types.is_datetime64_any_dtype(data[x_col])

    # ---- Libellés de l'axe X ----
    if month_col and x_col == month_col and year_col and data[year_col].nunique() > 1:
        data = data.sort_values([year_col, month_col])
        names = [
            f"{format_label(m)} {format_label(y)}"
            for m, y in zip(data[month_col], data[year_col])
        ]
    else:
        if time_like and _is_orderable(data[x_col]):
            data = data.sort_values(x_col)
        names = [format_label(v) for v in data[x_col]]

    # ---- Année(s) pour le titre ----
    raw_years = spec.get("years") or []
    if isinstance(raw_years, (int, str)):
        raw_years = [raw_years]
    years = list(dict.fromkeys(str(y).strip() for y in raw_years if str(y).strip().isdigit()))

    if not years and year_col and year_col in data.columns:
        years = sorted({format_label(v) for v in data[year_col].dropna()})

    year_already_visible = x_col == year_col or any(_YEAR_RE.search(n) for n in names)
    period_title = "" if year_already_visible else _period_suffix(years)

    first = y_cols[0]
    unit = detect_unit(first)
    hover_fmt = _hover_format(unit, [float(v) for v in data[first]])

    title = (
        f"{', '.join(pretty_column(c) for c in y_cols)} "
        f"par {pretty_column(x_col).lower()}"
        f"{period_title}"
    )
    x_title = ""

    # ---- Carte de France : axe région, une seule valeur par région ----
    if (
        requested in ("auto", "map", "line")
        and "region" in _norm(x_col)
        and len(y_cols) == 1
        and data[x_col].is_unique
    ):
        map_fig = _france_map(data, x_col, first, title, hover_fmt, pretty_column(first))
        if map_fig is not None:
            return map_fig

    # Une courbe n'a de sens que sur un axe temporel : sinon, barres
    if kind in ("auto", "map") or (kind == "line" and not time_like):
        kind = "line" if time_like else "bar"

    palette = [INDIGO, GOLD, "#3B82F6", "#0F766E", "#94A3B8"]
    fig = go.Figure()

    # ---- PIE ----
    if kind == "pie":
        pie = data.sort_values(first, ascending=False).head(8)
        fig.add_trace(
            go.Pie(
                labels=[format_label(v) for v in pie[x_col]],
                values=pie[first],
                hole=0.4,
                textinfo="label+percent",
            )
        )

    # ---- BAR : valeurs affichées, max en vert, min en rouge ----
    elif kind == "bar":
        if len(data) > 12 and len(y_cols) == 1:
            top = data.sort_values(first, ascending=False).head(15).iloc[::-1]
            vals = [float(v) for v in top[first]]
            fig.add_trace(
                go.Bar(
                    x=vals,
                    y=[format_label(v)[:40] for v in top[x_col]],
                    orientation="h",
                    marker_color=_extreme_colors(vals, palette[0]),
                    text=[format_value(v, unit, compact=True) for v in vals],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>{pretty_column(first)} : "
                        f"<b>%{{x:{hover_fmt}</b><extra></extra>"
                    ),
                )
            )
        else:
            for i, col in enumerate(y_cols):
                vals = [float(v) for v in data[col]]
                col_unit = detect_unit(col)
                fig.add_trace(
                    go.Bar(
                        x=names,
                        y=vals,
                        name=pretty_column(col),
                        marker_color=(
                            _extreme_colors(vals, palette[0])
                            if len(y_cols) == 1
                            else palette[i % len(palette)]
                        ),
                        text=(
                            [format_value(v, col_unit, compact=True) for v in vals]
                            if len(vals) <= 15 else None
                        ),
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            f"<b>%{{x}}</b><br>{pretty_column(col)} : "
                            f"<b>%{{y:{hover_fmt}</b><extra></extra>"
                        ),
                    )
                )

    # ---- LINE : points max/min colorés ----
    else:
        for i, col in enumerate(y_cols):
            color = palette[i % len(palette)]
            vals = [float(v) for v in data[col]]
            fig.add_trace(
                go.Scatter(
                    x=names,
                    y=vals,
                    name=pretty_column(col),
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    marker=dict(
                        size=8,
                        color=_extreme_colors(vals, color) if len(y_cols) == 1 else color,
                    ),
                    fill="tozeroy" if len(y_cols) == 1 else None,
                    fillcolor="rgba(79,70,229,0.12)",
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>{pretty_column(col)} : "
                        f"<b>%{{y:{hover_fmt}</b><extra></extra>"
                    ),
                )
            )

        matches = [a for a in (anomalies or []) if a.get("periode") in names]
        if matches and len(y_cols) == 1:
            fig.add_trace(
                go.Scatter(
                    x=[a["periode"] for a in matches],
                    y=[a["valeur"] for a in matches],
                    mode="markers",
                    name="Anomalie",
                    marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="#dc2626", width=3)),
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>Anomalie : <b>%{{y:{hover_fmt}</b><extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color=TEXT_DARK)),
        xaxis=dict(title=dict(text=x_title, font=dict(size=12)), type="category", tickangle=-45),
        yaxis=dict(title=None, tickformat=","),
        hovermode="closest" if kind in ("bar", "pie") else "x unified",
        margin=dict(l=60, r=30, t=70, b=90),
        height=340,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=len(y_cols) > 1 or bool(anomalies),
        barmode="group",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")
    return fig


def _period_suffix(years: list[str]) -> str:
    """['2024'] -> ' — 2024' ; ['2023','2024'] -> ' — 2023, 2024' ; 2022..2024 -> ' — 2022–2024'."""
    nums = sorted({int(y) for y in years})
    if not nums:
        return ""
    if len(nums) > 2 and nums[-1] - nums[0] == len(nums) - 1:
        return f" — {nums[0]}–{nums[-1]}"
    return " — " + ", ".join(str(n) for n in nums)


def _years_from(*texts: str | None) -> list[str]:
    """Années explicites : le premier texte qui en contient gagne (SQL avant question)."""
    for text in texts:
        found = sorted(set(_YEAR_RE.findall(text or "")))
        if found:
            return found
    return []



# ---------------------------------------------------------
# Prévision : tableau, KPI, panneau
# ---------------------------------------------------------

def make_forecast_chart(fr) -> go.Figure | None:
    """Graphique de prévision interactif : historique, prévision, zone d'incertitude."""
    try:
        date_col, value_col = fr.date_column, fr.value_column
        history = fr.historical[[date_col, value_col]].copy()
        forecast = fr.forecast.copy()
        history[date_col] = pd.to_datetime(history[date_col])
        forecast[date_col] = pd.to_datetime(forecast[date_col])

        unit = detect_unit(value_col)
        values = [*history[value_col].tolist(), *forecast[value_col].tolist()]
        hover = _hover_format(unit, values)

        fig = go.Figure()
        if {"Lower", "Upper"} <= set(forecast.columns):
            fig.add_trace(go.Scatter(
                x=[*forecast[date_col], *forecast[date_col][::-1]],
                y=[*forecast["Upper"], *forecast["Lower"][::-1]],
                fill="toself", fillcolor="rgba(201,162,39,0.18)", line=dict(width=0),
                hoverinfo="skip", name="Zone d'incertitude",
            ))
        fig.add_trace(go.Scatter(
            x=history[date_col], y=history[value_col], mode="lines+markers", name="Historique",
            line=dict(color=INDIGO, width=2.6), marker=dict(size=5), hovertemplate=f"%{{y:{hover}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=forecast[date_col], y=forecast[value_col], mode="lines+markers", name="Prévision",
            line=dict(color=GOLD, width=3, dash="dash"), marker=dict(size=6), hovertemplate=f"%{{y:{hover}<extra></extra>",
        ))
        if not history.empty:
            fig.add_vline(x=history[date_col].max(), line=dict(color="#94a3b8", dash="dot", width=1))

        fig.update_layout(
            title=dict(text=f"{pretty_column(value_col)} — historique et prévision", x=0,
                       xanchor="left", font=dict(size=18, color=TEXT_DARK)),
            plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
            margin=dict(l=20, r=20, t=70, b=30),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")
        return fig
    except Exception:
        logger.exception("Génération du graphique de prévision impossible")
        return None


RELIABILITY_BADGES = {
    "bonne": ("green", "✅"),
    "moyenne": ("orange", "⚠️"),
    "faible": ("red", "⛔"),
}


def month_label(value: Any) -> str:
    """2026-10-01 -> octobre 2026."""
    ts = pd.Timestamp(value)
    return f"{MONTHS_FR[ts.month - 1]} {ts.year}"


def same_period_last_year(fr) -> float | None:
    """Total de l'historique sur les mêmes mois un an plus tôt."""
    history = fr.historical.copy()
    history[fr.date_column] = pd.to_datetime(history[fr.date_column])
    series = history.set_index(fr.date_column)[fr.value_column]

    previous = [
        pd.Timestamp(d) - pd.DateOffset(years=1)
        for d in pd.to_datetime(fr.forecast[fr.date_column])
    ]

    if not all(p in series.index for p in previous):
        return None

    total = float(series.loc[previous].sum())
    return total if total > 0 else None


def forecast_table(fr) -> pd.DataFrame:
    """Tableau métier : historique + prévisions."""
    date_col, value_col = fr.date_column, fr.value_column

    history = fr.historical[[date_col, value_col]].copy()
    history.columns = ["Date", value_col]
    history["Type"] = "Historique"

    projected = pd.DataFrame(
        {
            "Date": fr.forecast[date_col],
            value_col: fr.forecast[value_col],
            "Type": "Prévision",
        }
    )

    table = pd.concat([history, projected], ignore_index=True)
    first = ["Date", "Type"]
    return table[first + [c for c in table.columns if c not in first]]


def build_forecast_reply(
    result,
    panel: dict[str, Any],
    elapsed: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Réponse métier + panneau pour un résultat de prévision."""
    fr = result.forecast_result

    date_col, value_col = fr.date_column, fr.value_column
    unit = detect_unit(value_col)
    metric = pretty_column(value_col).lower()

    forecast = fr.forecast
    count = len(forecast)
    total = float(forecast[value_col].sum())

    previous = same_period_last_year(fr)
    variation = (total / previous - 1) * 100 if previous else None

    reliability = getattr(fr, "reliability", "faible")
    css, icon = RELIABILITY_BADGES.get(reliability, ("", "ℹ️"))
    metrics = fr.metrics or {}
    mape = metrics.get("MAPE")
    reason = getattr(fr, "selection_reason", "") or ""
    folds = getattr(fr, "backtest_folds", 0)
    window = getattr(fr, "backtest_horizon", 0)

    notes = list(
        dict.fromkeys(
            [
                *(getattr(fr, "quality_warnings", None) or []),
                *(getattr(result, "warnings", None) or []),
            ]
        )
    )

    if variation is None:
        variation_text = ""
    elif round(variation) == 0:
        variation_text = "stable"
    else:
        variation_text = f"{variation:+.0f} %"

    sub = f"{variation_text} vs mêmes mois un an plus tôt" if variation_text else ""
    mape_text = f"MAPE {mape:.1f} %".replace(".", ",") if mape is not None else "non validée"

    badges = [
        '<span class="kpi-badge">📈 Prévision</span>',
        f'<span class="kpi-badge {css}">{icon} Fiabilité {reliability}</span>',
    ]
    if folds:
        badges.append(
            f'<span class="kpi-badge">🧪 Validée sur {folds} fenêtres de {window} mois</span>'
        )
    badges.append('<span class="kpi-badge green">🔒 Requête validée · lecture seule</span>')
    if elapsed is not None:
        seconds = f"{elapsed:.1f}".replace(".", ",")
        badges.append(f'<span class="kpi-badge">⏱ {seconds} s</span>')


    panel["kpi"] = (
        '<div class="kpi-section">'
        f'<div class="kpi-status-row">{"".join(badges)}</div>'
        "</div>"
    )

    forecast_chips = [
        (f"{pretty_column(value_col)} prévu · {count} mois",
         format_value(total, unit, compact=True) + (f" · {sub}" if sub else "")),
        ("Fiabilité", f"{reliability.capitalize()} · {mape_text}"),
        ("Modèle retenu", fr.model_name),
    ]
    panel["highlights"] = update(
        value=(
            '<div class="hl-band"><span class="hl-label">À retenir</span>'
            + "".join(
                f'<div class="hl-chip"><span class="hl-title">{html.escape(t)}</span>'
                f'<span class="hl-text">{html.escape(x)}</span></div>'
                for t, x in forecast_chips
            )
            + "</div>"
        ),
        visible=True,
    )

    # ---- Message de repli (l'assistant rédige le sien quand il est disponible) ----
    dates = pd.to_datetime(forecast[date_col])
    period = (
        f"{month_label(dates.min())} à {month_label(dates.max())}"
        if count > 1
        else month_label(dates.iloc[0])
    )

    headline = f"**{format_value(total, unit, compact=True)}** prévus sur {count} mois"
    if variation_text == "stable":
        headline += ", soit un niveau **stable** par rapport aux mêmes mois un an plus tôt"
    elif variation_text:
        headline += f", soit **{variation_text}** par rapport aux mêmes mois un an plus tôt"

    peak = forecast.loc[forecast[value_col].idxmax()]
    hist_dates = pd.to_datetime(fr.historical[fr.date_column])

    lines = [
        f"📈 **Prévision — {metric}** · {period}",
        f"{headline}.",
        f"Mois le plus fort : **{month_label(peak[date_col])}** "
        f"({format_value(peak[value_col], unit, compact=True)}).",
        f"{icon} **Fiabilité {reliability}**",
    ]
    if reason:
        lines.append(f"🧠 {reason}")
    if reliability == "faible":
        lines.append("Prenez ces chiffres avec prudence.")
    lines.append(
        f"Historique utilisé : {month_label(hist_dates.min())} → "
        f"{month_label(hist_dates.max())} ({len(hist_dates)} mois)."
    )

    message = "\n\n".join(lines)
    if notes:
        message += "\n\n" + "\n".join(f"- ⚠️ {n}" for n in notes)
    message += "\n\n> Détail mois par mois dans le tableau, graphique dans **Visualisation**."

    # ---- Tableau, exports, graphique ----
    table = clean_dataframe(forecast_table(fr))

    panel["table"] = update(value=table_payload(table), visible=True)
    panel["download"] = update(value=export_csv(table), visible=True)

    excel_path = export_excel(table)
    panel["excel"] = update(value=excel_path, visible=bool(excel_path))

    chart_fig = make_forecast_chart(fr)
    if chart_fig is not None:
        panel["chart"] = update(value=chart_fig, visible=True)

    # ---- Traçabilité : modèles comparés ----
    trace = build_trace(result, elapsed)

    ranking = getattr(fr, "candidates", None) or []
    if ranking:
        rows = "\n".join(
            f"| {r['model']} | {format_value(r['MAE'], unit)} | {r['MAPE']:.1f} % |".replace(".", ",")
            for r in ranking
        )
        trace += (
            "\n\n**Modèles comparés (backtest)**\n\n"
            "| Modèle | Erreur moyenne (MAE) | MAPE |\n"
            "|---|---:|---:|\n" + rows
        )

    panel["trace"] = trace

    trace_text = panel.get("trace", "")

    if isinstance(trace_text, str):
        panel["glossary"] = build_dynamic_glossary(trace_text)
    else:
        panel["glossary"] = build_dynamic_glossary("")

    if notes:
        panel["warnings"] = "\n".join(f"- ⚠️ {n}" for n in notes)

    return message, panel


# ---------------------------------------------------------
# Construction des sorties
# ---------------------------------------------------------



def empty_panel() -> dict[str, Any]:
    hidden = update(value=None, visible=False)

    return {
        "kpi": "",
        "table": hidden,
        "chart": hidden,
        "sql": "",
        "trace": "",
        "warnings": "",
        "download": hidden,
        "excel": hidden,
        "glossary": update(value=EMPTY_GLOSSARY, visible=True),
        "highlights": update(value="", visible=False),
        "plain": "",
        "pptx": hidden,
        "suggestions": [],
        "alerts": None,   # inchangé : seule la création d'une alerte le modifie
    }



def extract_requested_year(question: str) -> int | None:
    match = re.search(r"\b((?:19|20)\d{2})\b", question)
    return int(match.group(1)) if match else None


def build_no_data_message(result) -> str:
    """Message naturel lorsqu'aucune donnée n'est trouvée (pipeline classique)."""
    question = getattr(result, "question", "") or ""

    requested_year = extract_requested_year(question)
    max_year = getattr(result, "data_max_year", None)

    if requested_year is not None and max_year is not None and requested_year > max_year:
        return (
            f"Je n’ai pas trouvé d’information concernant votre demande "
            f"pour **{requested_year}**, car les données disponibles "
            f"s’arrêtent en **{max_year}**."
        )

    if requested_year is not None:
        return (
            f"Je n’ai trouvé aucune information correspondant à votre "
            f"demande pour **{requested_year}**."
        )

    return (
        "Je n’ai trouvé aucune information correspondant à votre demande."
        "\n\nVous pouvez essayer avec une autre période ou un autre critère."
    )


def prepare_dataframe_for_display(df):
    """Noms techniques remplacés par des noms naturels pour l'affichage."""
    if df is None:
        return df

    df = df.copy()
    df = df.drop(columns=[c for c in ["Nom_Mois"] if c in df.columns], errors="ignore")

    return df.rename(
        columns={
            "Annee": "Année",
            "Mois": "Mois",
            "Chiffre_Affaires": "Chiffre d’affaires",
        }
    )


def clean_chat_analysis(text: str) -> str:
    """Nettoie la réponse LLM destinée au chatbot.

    Le tableau détaillé est affiché séparément dans le panneau Résultats.
    """
    if not text:
        return ""

    text = text.strip()

    # Supprime les blocs de tableaux Markdown
    lines = text.splitlines()
    cleaned = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Détection d'une ligne de tableau Markdown
        is_table_line = (
            stripped.startswith("|")
            and stripped.endswith("|")
        )

        if is_table_line:
            in_table = True
            continue

        # Ligne de séparation Markdown : |---|---:|
        if in_table and re.fullmatch(r"\|?[\s:\-|]+\|?", stripped):
            continue

        if in_table:
            # Une ligne normale termine le tableau
            in_table = False

        cleaned.append(line)

    text = "\n".join(cleaned)

    # Nettoyage des lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ---------------------------------------------------------
# Bande « À retenir »
# ---------------------------------------------------------


def build_highlights(df: pd.DataFrame, meta: dict, extra: list | None = None) -> str:
    """Plus forte hausse / baisse / anomalies, calculées en code."""
    measures = measure_columns(df)
    if not measures or len(df) < 2:
        return ""

    col = measures[0]
    unit = detect_unit(col)

    def fmt(v: Any) -> str:
        return format_value(v, unit, compact=True)

    def pct(r: float) -> str:
        return f"{r * 100:+.1f} %".replace(".", ",")

    chips: list[tuple[str, str, str]] = list(extra or [])

    cmp_ = meta.get("compare") or {}

    if is_additive(col, df[col].dropna(), unit):
        total_text = fmt(df[col].sum())
        if cmp_.get("pct") is not None and _norm(cmp_.get("measure")) == _norm(col):
            total_text += " · " + delta_html(cmp_["pct"], cmp_.get("label", "")) \
                .replace("<span", "").replace("</span>", "")  # texte simple
        chips.append(("", "Total", total_text))

    # cmp_ = meta.get("compare") or {}
    by_label = cmp_.get("by_label") or {}

    if by_label:
        k, v = max(by_label.items(), key=lambda kv: kv[1])
        if v > 0:
            chips.append(("up", "▲ Plus forte hausse", f"{k} · {pct(v)} {cmp_.get('label', '')}"))
        k, v = min(by_label.items(), key=lambda kv: kv[1])
        if v < 0:
            chips.append(("down", "▼ Plus forte baisse", f"{k} · {pct(v)} {cmp_.get('label', '')}"))
    else:
        points = _ordered_points(df, col)
        if points and len(points) >= 3:
            moves = [
                (f"{a[0]} → {b[0]}", b[1] / a[1] - 1)
                for a, b in zip(points, points[1:])
                if a[1]
            ]
            if moves:
                name, v = max(moves, key=lambda m: m[1])
                if v > 0:
                    chips.append(("up", "▲ Plus forte hausse", f"{name} · {pct(v)}"))
                name, v = min(moves, key=lambda m: m[1])
                if v < 0:
                    chips.append(("down", "▼ Plus forte baisse", f"{name} · {pct(v)}"))
        else:
            data = df.dropna(subset=[col])
            labels = label_columns(df)
            if labels and len(data) >= 2:
                hi, lo = data.loc[data[col].idxmax()], data.loc[data[col].idxmin()]
                chips.append(("up", "▲ Le plus élevé", f"{format_label(hi[labels[0]])} · {fmt(hi[col])}"))
                chips.append(("down", "▼ Le plus faible", f"{format_label(lo[labels[0]])} · {fmt(lo[col])}"))

    for a in (meta.get("anomalies") or [])[:2]:
        gap = a.get("ecart_pct")
        detail = f" de {abs(gap):.0f} %".replace(".", ",") if gap is not None else ""
        chips.append(
            ("warn", "⚠ Valeur anormale", f"{a.get('periode')} · {a.get('sens', 'écart')}{detail} vs attendu")
        )

    if not chips:
        return ""

    body = "".join(
        f'<div class="hl-chip {css}"><span class="hl-title">{html.escape(t)}</span>'
        f'<span class="hl-text">{html.escape(x)}</span></div>'
        for css, t, x in chips[:4]
    )
    return f'<div class="hl-band"><span class="hl-label">À retenir</span>{body}</div>'


# ---------------------------------------------------------
# Export PowerPoint : une diapositive (titre, graphique, 3 phrases)
# ---------------------------------------------------------


def _chart_points(df: pd.DataFrame, spec: dict):
    measures, labels = measure_columns(df), label_columns(df)
    if not measures or not labels:
        return None

    y = next(
        (c for c in (_resolve_column(df, p) for p in str(spec.get("y") or "").split(",")) if c in measures),
        measures[0],
    )
    points = _ordered_points(df, y)
    if points:
        names, values = zip(*points)
        return y, list(names), [float(v) for v in values], "line"

    x = _resolve_column(df, spec.get("x")) or labels[0]
    top = df.dropna(subset=[y]).sort_values(y, ascending=False).head(10)
    return y, [format_label(v) for v in top[x]], [float(v) for v in top[y]], "bar"


def _sentences(text: str, n: int = 3) -> list[str]:
    t = re.sub(r"\[\[SUITES\]\].*", "", text or "", flags=re.S)
    t = re.sub(r"[*_`>#]", "", t)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", t)]
    return [p for p in parts if len(p) > 25][:n]


INDIGO_RGB = "1E3A6E"


def _pptx_base_slide(prs, title: str, data_until=None):
    """Diapositive vierge avec bandeau, titre et pied de page ; renvoie (slide, add_text)."""
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    slide = prs.slides.add_slide(prs.slide_layouts[6])

    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.18))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor.from_string(INDIGO_RGB)
    band.line.fill.background()

    def add_text(left, top, width, height, lines, size, bold=False, color="1E1B4B"):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.space_after = Pt(10)

    add_text(0.5, 0.4, 12.3, 0.9, [title or "Analyse"], 28, bold=True)

    footer = "Assistant Data AI"
    if data_until:
        footer += f" · Données jusqu'à {data_until}"
    footer += f" · Généré le {time.strftime('%d/%m/%Y')}"
    add_text(0.5, 6.9, 12.3, 0.4, [footer], 10, color="64748B")
    return slide, add_text


def _pptx_new_presentation():
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    return prs


def _pptx_chart_slide(prs, df, spec, title, text, data_until=None) -> bool:
    """Ajoute une diapositive graphique natif + « À retenir ». False si pas de graphique possible."""
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches, Pt

    points = _chart_points(df, spec or {})
    if points is None:
        return False
    value_col, names, values, kind = points

    slide, add_text = _pptx_base_slide(prs, title, data_until)
    indigo = RGBColor.from_string(INDIGO_RGB)

    cd = CategoryChartData()
    cd.categories = names
    cd.add_series(pretty_column(value_col), values)
    ctype = XL_CHART_TYPE.LINE_MARKERS if kind == "line" else XL_CHART_TYPE.COLUMN_CLUSTERED
    chart = slide.shapes.add_chart(
        ctype, Inches(0.5), Inches(1.5), Inches(8.2), Inches(5.0), cd
    ).chart
    chart.has_legend = False
    chart.has_title = False
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.value_axis.tick_labels.font.size = Pt(10)

    plot = chart.plots[0]
    plot.has_data_labels = len(names) <= 12
    if plot.has_data_labels:
        plot.data_labels.font.size = Pt(10)
        plot.data_labels.number_format = "#,##0"
        plot.data_labels.number_format_is_linked = False

    series = plot.series[0]
    if kind == "line":
        series.format.line.color.rgb = indigo
        series.smooth = False
    else:
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = indigo

    add_text(9.0, 1.5, 3.9, 0.5, ["À retenir"], 16, bold=True, color=INDIGO_RGB)
    add_text(9.0, 2.1, 3.9, 4.3, _sentences(text, 3), 14)
    return True


def _pptx_text_slide(prs, title, lines, data_until=None) -> None:
    _, add_text = _pptx_base_slide(prs, title, data_until)
    add_text(0.7, 1.5, 11.9, 5.2, [f"• {line}" for line in lines] or [" "], 18)


def export_pptx(df, spec, title, text, data_until=None) -> str | None:
    """Diapositive 16:9 avec graphique natif (modifiable dans PowerPoint)."""
    try:
        prs = _pptx_new_presentation()
    except ImportError:
        logger.warning("Export PowerPoint indisponible (pip install python-pptx)")
        return None

    try:
        if not _pptx_chart_slide(prs, df, spec, title, text, data_until):
            return None
        path = Path(tempfile.mkdtemp()) / "analyse.pptx"
        prs.save(path)
        return str(path)

    except Exception:
        logger.exception("Export PowerPoint impossible")
        return None


# ---------------------------------------------------------
# Alertes, filtres, pastilles de questions de suite
# ---------------------------------------------------------


def alerts_update(triggered: list[dict]):
    if not triggered:
        return update(value="", visible=False)

    items = ""
    for a in triggered:
        more = f" (+{a['count'] - len(a['sample'])} autres)" if a["count"] > len(a["sample"]) else ""
        detail = " ; ".join(html.escape(s) for s in a["sample"])
        items += (
            f"<li><b>{html.escape(a['name'])}</b> — {html.escape(a['message'])} : {detail}{more}</li>"
        )

    n = len(triggered)
    return update(
        value=(
            f'<div class="alert-banner"><div class="alert-title">🔔 {n} alerte{"s" if n > 1 else ""} active{"s" if n > 1 else ""}'
            f"</div><ul>{items}</ul></div>"
        ),
        visible=True,
    )








def build_reply(
    result,
    elapsed: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Transforme le résultat (agent ou pipeline classique) en réponse + panneau."""
    panel = empty_panel()

    sql = getattr(result, "sql", None)
    if sql:
        panel["sql"] = sql

    if result.status == "clarification_needed":
        question = result.clarification_question or "Pouvez-vous préciser votre demande ?"
        return (
            "🤔 **Pour être sûr de bien vous répondre, "
            f"j’ai besoin d’une précision.**\n\n{question}",
            panel,
        )

    if result.status == "validation_error":
        return (
            "Je n’ai pas pu traiter cette demande de manière fiable. "
            "Pouvez-vous reformuler votre question ou préciser "
            "les informations que vous recherchez ?",
            panel,
        )

    if result.status == "error":
        detail = (getattr(result, "errors", None) or [""])[0]
        message = "Je n’ai pas pu traiter cette demande."
        if detail:
            message += f"\n\n{detail}"
        return message, panel

    if result.status != "ok":
        logger.warning("État inattendu retourné par le pipeline : %s", result.status)
        return (
            "Je rencontre un petit problème pour traiter cette demande. "
            "Vous pouvez réessayer dans un instant.",
            panel,
        )

    if result.warnings:
        panel["warnings"] = "\n".join(f"- ⚠️ {w}" for w in result.warnings)

    analysis_text = getattr(result, "analysis_text", None)
    meta = getattr(result, "metadata", None) or {}

    # ---- Éléments communs : pastilles, phrase en langage clair, alertes ----
    panel["suggestions"] = [s for s in (meta.get("suggestions") or []) if s][:3]

    plain = meta.get("plain") or []
    if plain:
        panel["plain"] = "💬 " + " ".join(plain)

    if meta.get("alert_created"):
        try:
            panel["alerts"] = alerts_update(get_pipeline().get_agent().check_alerts())
        except Exception:
            logger.exception("Rafraîchissement des alertes impossible")

    # ---- Prévision ----
    if getattr(result, "forecast_result", None) is not None:
        message, panel = build_forecast_reply(result, panel, elapsed)
        final_message = clean_chat_analysis(analysis_text) if analysis_text else message
        panel["glossary"] = update(
            value=build_dynamic_glossary(f"{panel['trace']}\n{final_message}\n{sql or ''}"),
            visible=True,
        )
        return final_message, panel

    # ---- Données ----
    dataframe = result.dataframe
    if dataframe is not None and not dataframe.empty:
        dataframe = dataframe.dropna(how="all")

    if dataframe is None or dataframe.empty:
        if analysis_text:
            return clean_chat_analysis(analysis_text), panel
        return build_no_data_message(result), panel

    df = clean_dataframe(dataframe)
    rows = len(df)
    question = getattr(result, "question", "") or ""

    if analysis_text:
        message = clean_chat_analysis(analysis_text)
    else:
        answer = build_answer(question, df)
        message = answer or "Voici ce que j’ai trouvé."
        insights = build_insights(question, df) if rows > 1 else []
        if insights:
            message += "\n\n**📌 L'essentiel**\n" + "\n".join(f"- {line}" for line in insights)
        if rows > 1 or not answer:
            message += f"\n\n> {plural(rows, 'résultat')} — détail dans le tableau."

    panel["kpi"] = build_kpi_html(
        df,
        elapsed,
        question,
        last_updated=getattr(result, "last_updated", None),
        compare=meta.get("compare"),
        data_until=meta.get("data_until"),
    )

    display_df = prepare_dataframe_for_display(df)
    panel["table"] = update(value=table_payload(display_df), visible=True)
    panel["download"] = update(value=export_csv(display_df), visible=True)

    excel_path = export_excel(display_df)
    panel["excel"] = update(value=excel_path, visible=bool(excel_path))

    chart_spec = dict(meta.get("chart") or {})
    if not chart_spec.get("years"):
        chart_spec["years"] = _years_from(getattr(result, "sql", None), question)

    focus = meta.get("focus_region")
    regions_df = meta.get("regions_df")

    chart, extra = None, []
    if focus and regions_df is not None:
        rdf = clean_dataframe(regions_df)
        chart = _focus_map(rdf, focus, chart_spec["years"])
        chip = _rank_chip(rdf, focus)
        if chip:
            extra.append(chip)

    if chart is None:
        chart = make_chart_from_spec(df, chart_spec, meta.get("anomalies"))

    panel["chart"] = update(value=chart, visible=chart is not None)

    highlights = build_highlights(df, meta, extra)
    panel["highlights"] = update(value=highlights, visible=bool(highlights))

    slide_title = chart.layout.title.text if chart is not None else question
    pptx_path = export_pptx(df, chart_spec, slide_title, message, meta.get("data_until"))
    panel["pptx"] = update(value=pptx_path, visible=bool(pptx_path))

    panel["trace"] = build_trace(result, elapsed)
    panel["glossary"] = update(
        value=build_dynamic_glossary(f"{panel['trace']}\n{message}\n{sql or ''}"),
        visible=True,
    )

    return message, panel


# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------














def _match_region(name: Any) -> str | None:
    """Nom reconnu parmi les régions proposées dans le filtre (ou None)."""
    if not name:
        return None
    try:
        options = get_pipeline().get_agent().filter_options().get("region", [])
    except Exception:
        return None
    key = _plain_key(name)
    return next((o for o in options if _plain_key(o) == key), None)


ZOOM_PREFIX = "Détail du chiffre d'affaires de la région"












GLOSSARY = {
    "MAPE": (
        "Erreur moyenne en pourcentage entre les valeurs réelles "
        "et les prévisions. Plus elle est faible, mieux c'est."
    ),
    "MAE": (
        "Erreur absolue moyenne entre les valeurs réelles "
        "et les prévisions. Plus elle est faible, mieux c'est."
    ),
    "RMSE": (
        "Erreur qui pénalise davantage les grosses erreurs. "
        "Plus elle est faible, mieux c'est."
    ),
    "XGBoost": (
        "Algorithme de machine learning basé sur des arbres de décision, "
        "capable de détecter des relations complexes dans les données."
    ),
    "Holt-Winters": (
        "Méthode statistique de prévision qui prend en compte "
        "la tendance et la saisonnalité."
    ),
    "Holt amorti": (
        "Méthode de prévision qui modélise la tendance "
        "en réduisant progressivement son influence dans le futur."
    ),
    "Régression linéaire": (
        "Méthode statistique qui estime une tendance entre "
        "une variable à prévoir et des variables explicatives."
    ),
    "Régression linéaire saisonnière": (
        "Régression linéaire qui tient également compte "
        "des variations saisonnières."
    ),
    "Saisonnalité": (
        "Variation qui se répète régulièrement dans le temps, "
        "par exemple chaque mois ou chaque année."
    ),
    "Backtesting": (
        "Technique consistant à tester un modèle sur des périodes "
        "historiques connues afin d'évaluer ses performances."
    ),
    "Forecast": (
        "Prévision des valeurs futures à partir des données historiques."
    ),
    "Prévision": (
        "Estimation de valeurs futures à partir de l'historique disponible."
    ),
    "Modèle sélectionné": (
        "Modèle ayant obtenu les meilleures performances "
        "lors des tests historiques."
    ),
}


BUSINESS_GLOSSARY = {
    "Chiffre d'affaires": "Somme des montants des ventes sur la période (colonne Chiffre_Affaires de Fact_Ventes).",
    "CA": "Abréviation de chiffre d'affaires.",
    "Marge": "Somme des marges des ventes sur la période (colonne Marge de Fact_Ventes).",
    "Taux de marge": "Marge divisée par le chiffre d'affaires de la même période, exprimée en %.",
    "Quantité": "Nombre d'unités vendues (colonne Quantite de Fact_Ventes).",
    "Panier moyen": "Chiffre d'affaires divisé par le nombre de ventes.",
    "N-1": "Même période de l'année précédente, utilisée comme base de comparaison.",
    "Anomalie": (
        "Valeur statistiquement inhabituelle compte tenu de la tendance "
        "et de la saisonnalité (écart supérieur à 3 écarts robustes)."
    ),
    "Tendance": "Évolution moyenne par mois, calculée après neutralisation de la saisonnalité.",
}
GLOSSARY.update(BUSINESS_GLOSSARY)


def build_dynamic_glossary(text: str) -> str:
    def empty(msg: str) -> str:
        return f'<div class="dynamic-glossary-empty">{msg}</div>'

    if not text:
        return empty("Les termes utilisés dans la réponse apparaîtront ici.")

    text = text.replace("’", "'")
    found = []

    for term, definition in GLOSSARY.items():
        flags = 0 if len(term) <= 3 else re.IGNORECASE   # « CA », « MAE » : casse stricte
        m = re.search(rf"\b{re.escape(term)}\b", text, flags)
        if m:
            found.append((m.start(), m.end(), term, definition))

    # « Marge » est masqué quand « Taux de marge » le contient
    found = [
        f for f in found
        if not any(
            o is not f and o[0] <= f[0] and o[1] >= f[1] and (o[1] - o[0]) > (f[1] - f[0])
            for o in found
        )
    ]
    found.sort(key=lambda x: x[0])

    if not found:
        return empty("Aucun terme technique ou métier détecté.")

    blocks = "".join(
        f'<div class="glossary-item"><div class="glossary-term">{html.escape(term)}</div>'
        f'<div class="glossary-definition">{html.escape(definition)}</div></div>'
        for _, _, term, definition in found
    )
    return f'<div class="dynamic-glossary">{blocks}</div>'


# ---------------------------------------------------------
# Agent analyste : mode Mission
# ---------------------------------------------------------

MISSION_MAX_STEPS = 6

MISSION_EXAMPLES = [
    "Pourquoi la marge évolue-t-elle ainsi cette année ?",
    "Fais le bilan commercial complet de la dernière année disponible",
    "Quelles régions et quels produits faut-il prioriser ?",
    "Identifie les anomalies et les risques dans les ventes récentes",
]

MISSION_PHASES = {
    "plan": "🧭 Élaboration du plan d'analyse…",
    "report": "✍️ Rédaction du rapport…",
}

STEP_ICONS = {"pending": "⚪", "running": "⏳", "done": "✅", "error": "⚠️"}


def mission_progress_html(snap: dict[str, Any], elapsed: float | None = None) -> str:
    steps = snap.get("steps") or []
    phase = snap.get("phase", "plan")
    done = sum(1 for s in steps if s[1] in ("done", "error"))

    if phase == "done":
        status = f"✅ Mission terminée en {elapsed:.0f} s" if elapsed else "✅ Mission terminée"
    elif phase == "error":
        status = "⚠️ Mission interrompue"
    elif phase == "steps":
        status = f"🔎 Analyses en cours ({done}/{len(steps)})"
    else:
        status = MISSION_PHASES.get(phase, "")

    items = ""
    for title, state, activity, finding in steps:
        detail = ""
        if state == "running" and activity:
            detail = f'<div class="mission-activity">{html.escape(activity)}…</div>'
        elif state in ("done", "error") and finding:
            short = finding if len(finding) <= 220 else finding[:217].rsplit(" ", 1)[0] + "…"
            detail = f'<div class="mission-finding">{html.escape(short)}</div>'
        items += (
            f'<li class="mission-step {state}"><span class="mission-icon">{STEP_ICONS.get(state, "⚪")}</span>'
            f"<div><b>{html.escape(title)}</b>{detail}</div></li>"
        )

    title = html.escape(snap.get("title") or "")
    return (
        '<div class="mission-progress">'
        + (f'<div class="mission-title">🎯 {title}</div>' if title else "")
        + f'<div class="mission-status">{status}</div>'
        + (f'<ol class="mission-steps">{items}</ol>' if items else "")
        + "</div>"
    )


def _report_section(report: str, name: str) -> list[str]:
    """Lignes d'une section « ## name » du rapport Markdown (puces nettoyées)."""
    match = re.search(rf"^##\s*{name}[^\n]*\n(.*?)(?=^##\s|\Z)", report or "", re.S | re.M | re.I)
    if not match:
        return []
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).replace("**", "").strip()
             for ln in match.group(1).splitlines()]
    return [ln for ln in lines if ln]


def _step_chart(step):
    if step.forecast_result is not None:
        try:
            return make_forecast_chart(step.forecast_result)
        except Exception:
            logger.exception("Graphique de prévision de mission impossible")
            return None
    df = step.dataframe
    if df is None or df.empty:
        return None
    try:
        return make_chart_from_spec(clean_dataframe(df), step.chart or {}, step.anomalies)
    except Exception:
        logger.exception("Graphique de mission impossible")
        return None


def _step_markdown(index: int, step) -> str:
    text = f"### {index}. {step.title}\n\n{step.finding or '—'}"
    plain = " ".join(q["plain"] for q in step.queries if q.get("plain"))
    if plain:
        text += f"\n\n💬 *{plain}*"
    sqls = "\n\n".join(q["sql"] for q in step.queries if q.get("sql"))
    if sqls:
        text += f"\n\n<details><summary>Requêtes SQL</summary><pre>{html.escape(sqls)}</pre></details>"
    return text


def export_mission_markdown(mission) -> str:
    parts = [f"# {mission.title or mission.objective}", f"*Mission : {mission.objective}*", mission.report]
    parts.append("# Détail des analyses")
    for i, step in enumerate(mission.steps, 1):
        parts.append(f"## {i}. {step.title}\n\n{step.finding}")
    path = Path(tempfile.mkdtemp()) / "rapport_mission.md"
    path.write_text("\n\n".join(p for p in parts if p), encoding="utf-8")
    return str(path)


def export_mission_excel(mission) -> str | None:
    try:
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        return None
    try:
        path = Path(tempfile.mkdtemp()) / "mission.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary = pd.DataFrame(
                {
                    "Étape": [f"{i}. {s.title}" for i, s in enumerate(mission.steps, 1)],
                    "Constat": [s.finding for s in mission.steps],
                }
            )
            summary.to_excel(writer, index=False, sheet_name="Synthèse")
            used = {"Synthèse"}
            for i, step in enumerate(mission.steps, 1):
                if step.dataframe is None or step.dataframe.empty:
                    continue
                name = re.sub(r"[\[\]:*?/\\]", " ", f"{i}. {step.title}")[:31].strip()
                while name in used:
                    name = name[:28] + f"~{i}"
                used.add(name)
                export_frame(clean_dataframe(step.dataframe)).to_excel(writer, index=False, sheet_name=name)
            for sheet in writer.sheets.values():
                for cell in sheet[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor=INDIGO_RGB)
                for column in sheet.columns:
                    longest = max(len(str(c.value or "")) for c in list(column)[:200])
                    sheet.column_dimensions[column[0].column_letter].width = min(longest + 3, 80)
        return str(path)
    except Exception:
        logger.exception("Export Excel de la mission impossible")
        return None


def export_mission_pptx(mission) -> str | None:
    """Deck : synthèse, une diapositive par analyse, recommandations."""
    try:
        prs = _pptx_new_presentation()
    except ImportError:
        return None
    try:
        until = mission.data_until
        synthesis = _report_section(mission.report, "Synthèse")
        _pptx_text_slide(prs, mission.title or mission.objective, synthesis or [mission.objective], until)
        key_facts = _report_section(mission.report, "Constats clés")
        if key_facts:
            _pptx_text_slide(prs, "Constats clés", key_facts, until)
        for i, step in enumerate(mission.steps, 1):
            df = step.dataframe
            title = f"{i}. {step.title}"
            if df is None or df.empty or not _pptx_chart_slide(
                prs, clean_dataframe(df), step.chart or {}, title, step.finding, until
            ):
                _pptx_text_slide(prs, title, _sentences(step.finding, 4), until)
        recos = _report_section(mission.report, "Recommandations")
        if recos:
            _pptx_text_slide(prs, "Recommandations", recos, until)
        path = Path(tempfile.mkdtemp()) / "mission.pptx"
        prs.save(path)
        return str(path)
    except Exception:
        logger.exception("Export PowerPoint de la mission impossible")
        return None









# Vues de l'application (l'accueil = le chat). L'attribut data-view posé sur l'en-tête
# sert uniquement à mettre en évidence le bouton de la vue active.













