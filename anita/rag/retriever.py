"""
High-level retrieval interface for the Recommendation Engine.

Combines metadata filtering (destination match -- exact, cheap, always
correct) with semantic search (interest/preference matching -- fuzzy,
where retrieval actually earns its keep). This two-stage pattern (filter
then rank) is standard RAG practice: never rely on semantic similarity
alone for a hard constraint like "must be in Agra, not Kerala."
"""

from __future__ import annotations

from rag.documents import DOCUMENTS, Document, documents_for_destination
from rag.vector_store import ScoredDocument, VectorStore

_store_cache: dict[str, VectorStore] = {}


def _get_store_for_destination(destination: str) -> VectorStore:
    """
    Build (or reuse) a per-destination vector store. Indexing per-destination
    rather than once globally keeps the TF-IDF vocabulary focused on that
    destination's actual content, and sidesteps any cross-destination score
    inflation from unrelated vocabulary in the shared fit.
    """
    key = destination.strip().lower()
    if key not in _store_cache:
        docs = documents_for_destination(destination)
        store = VectorStore()
        if docs:
            store.build_index(docs)
        _store_cache[key] = store
    return _store_cache[key]


def build_interest_query(interests: dict, hidden_preferences: dict, health: dict) -> str:
    """
    Turn a traveller_profile's interests/hidden_preferences/health into a
    natural-language-ish query string for semantic search. Weighted
    interests (score >= 6) and relevant health constraints are included;
    everything else is left out to keep the query focused on what
    actually matters for this traveller.
    """
    terms = []
    for interest, score in (interests or {}).items():
        if isinstance(score, (int, float)) and score >= 6:
            terms.append(interest)

    hidden = hidden_preferences or {}
    if (hidden.get("adventure_index") or 0) >= 6:
        terms.append("adventure")
    if (hidden.get("relaxation_index") or 0) >= 6:
        terms.append("relaxation")
    if (hidden.get("cultural_curiosity") or 0) >= 6:
        terms.append("culture history")
    if (hidden.get("food_explorer_score") or 0) >= 6:
        terms.append("food local experience")

    if (health or {}).get("walking_difficulty"):
        terms.append("low walking easy accessible")

    return " ".join(terms) if terms else "popular highlights"


def _live_poi_as_documents(destination: str, poi_names: list[str]) -> list[Document]:
    """
    Wrap real Google Places POI names (fetched during the map exploration
    flow, see streamlit_app.py's nearby_places_live) as lightweight
    Document objects so they can be retrieved alongside the curated
    corpus. These carry a distinct "live_places_data" tag and minimal
    text -- Places gives names/coordinates, not experiential descriptions
    -- so they'll generally score lower on semantic similarity than a
    well-written curated document, but they're grounded in this specific
    trip's actual locked location, which the static corpus can never be.
    """
    return [
        Document(
            id=f"live_poi_{i}",
            destination=destination,
            title=name,
            text=f"{name} -- a real point of interest near the traveller's locked stay location in {destination}.",
            tags=["live_places_data"],
        )
        for i, name in enumerate(poi_names)
    ]


def retrieve_activities(destination: str, interests: dict, hidden_preferences: dict,
                          health: dict, k: int = 5, live_poi_names: list[str] | None = None,
                          max_live_poi: int = 4) -> list[ScoredDocument]:
    """
    Retrieve the top-k most relevant grounded activity documents for a
    destination, ranked by semantic similarity to the traveller's
    interests. Returns [] if the destination isn't in the corpus AND no
    live_poi_names are given (the caller -- recommendation_engine.py's
    plan_activities -- should treat an empty result as "no grounded
    content available," not an error).

    `live_poi_names`, if provided (typically from
    profile.trip.stay_location.nearby_points_of_interest, i.e. real
    Google Places data fetched during the map exploration flow), are
    appended as supplementary candidates -- NOT blended into the same
    TF-IDF-ranked pool. Bare place names ("Shahi Jama Masjid Agra") share
    almost no literal vocabulary with a typical interest query ("history
    photography"), so lexical similarity scoring would near-always filter
    them out even when they're obviously relevant by name -- TF-IDF has
    no way to know a "Masjid" is historically/culturally significant the
    way an LLM reasoning over the name would. Since they're real,
    physically-relevant-by-proximity candidates regardless of semantic
    match, they're included unconditionally (up to max_live_poi) and left
    for the LLM judgment layer to actually reason about, or shown as-is
    in the no-LLM raw fallback.
    """
    curated_store = _get_store_for_destination(destination)
    query = build_interest_query(interests, hidden_preferences, health)
    curated_results = curated_store.search(query, k=k)

    if not live_poi_names:
        return curated_results

    live_docs = _live_poi_as_documents(destination, live_poi_names)[:max_live_poi]
    # No real TF-IDF score to assign -- these aren't ranked by lexical
    # similarity, they're included because they're real and nearby.
    # score=0.0 clearly signals "not semantically scored" to any caller
    # inspecting it, distinct from a genuine low-but-nonzero TF-IDF match.
    live_results = [ScoredDocument(document=d, score=0.0) for d in live_docs]

    return curated_results[:k] + live_results


def retrieve_for_query(destination: str, query: str, k: int = 5) -> list[ScoredDocument]:
    """Lower-level: retrieve by an arbitrary free-text query, for callers
    that want more direct control than build_interest_query provides."""
    store = _get_store_for_destination(destination)
    return store.search(query, k=k)


def available_destinations() -> list[str]:
    """Destinations currently covered by the knowledge base."""
    return sorted({d.destination for d in DOCUMENTS})
