"""
Tests for the RAG retrieval pipeline (rag/) and its integration into
recommendation_engine.py's plan_activities(). Run with:
    python tests/test_rag.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_semantic_ranking_is_sensible():
    """A history-focused query should rank history-tagged docs above
    unrelated ones for the same destination."""
    from rag.retriever import retrieve_for_query

    results = retrieve_for_query("Agra", "history culture monuments temples", k=3)
    assert len(results) > 0, "Expected at least one result"
    top_tags = set(results[0].document.tags)
    assert "history" in top_tags, f"Top result should be history-tagged, got tags={top_tags}"
    print("PASS: semantic ranking surfaces relevant tags for a topical query")


def test_destination_filtering():
    """Retrieval must never leak documents from a different destination."""
    from rag.retriever import retrieve_for_query

    results = retrieve_for_query("Agra", "anything at all", k=20)
    for r in results:
        assert r.document.destination == "Agra", f"Leaked non-Agra document: {r.document.destination}"
    print("PASS: destination filtering never leaks cross-destination documents")


def test_unknown_destination_returns_empty_not_crash():
    from rag.retriever import retrieve_for_query

    results = retrieve_for_query("Nowhereland", "anything", k=5)
    assert results == []
    print("PASS: unknown destination degrades to empty list, no crash")


def test_hard_filter_excludes_incompatible_activities():
    """walking_difficulty must exclude activities tagged as such, even
    though they might otherwise score highly on interest match."""
    from recommendation_engine import plan_activities
    from chat_assistant_prompt import new_traveller_profile

    profile = new_traveller_profile()
    profile["trip"]["destination"]["confirmed"] = ["Paris"]
    profile["interests"] = {"culture": 9}
    profile["health"]["walking_difficulty"] = True

    results = plan_activities(profile, llm_client=None)
    titles = [r["activity"]["title"] for r in results if r["activity"]]
    assert "Montmartre and Sacré-Cœur" not in titles, "Hard filter should exclude walking_difficulty-tagged activity"
    assert len(titles) > 0, "Other Paris activities should still be returned"
    print("PASS: hard health/accessibility filter correctly excludes incompatible retrieved activities")


def test_llm_judgment_layer_drops_fabricated_ids():
    """If the LLM judgment layer references a candidate id that was never
    actually offered to it, that item must be silently dropped -- never
    trusted. Same fabrication-protection principle as the Chat Assistant's
    traveller_composition.members guardrail, applied to this sub-engine."""
    from recommendation_engine import plan_activities
    from chat_assistant_prompt import new_traveller_profile

    class FabricatingJudgmentClient:
        def complete_json(self, system, user_content):
            request = json.loads(user_content)
            real_id = request["candidates"][0]["id"]
            return [
                {"id": real_id, "score": 9, "rationale": "Legitimate match."},
                {"id": "totally_made_up_id_not_in_candidates", "score": 10, "rationale": "Should be dropped."},
            ]

    profile = new_traveller_profile()
    profile["trip"]["destination"]["confirmed"] = ["Agra"]
    profile["interests"] = {"history": 9}

    results = plan_activities(profile, llm_client=FabricatingJudgmentClient())
    assert len(results) == 1, f"Expected only the legitimate id to survive, got {len(results)}"
    print("PASS: LLM judgment layer's fabricated candidate id was correctly dropped")


def test_no_destination_degrades_gracefully():
    from recommendation_engine import plan_activities
    from chat_assistant_prompt import new_traveller_profile

    profile = new_traveller_profile()  # no destination set at all
    results = plan_activities(profile, llm_client=None)
    assert results[0]["activity"] is None
    assert "destination" in results[0]["rationale"].lower()
    print("PASS: missing destination degrades gracefully with a clear rationale")


def test_live_places_and_curated_content_combine():
    """Real Google Places POI data (from the map exploration flow) must
    be combined with curated corpus content, not treated as redundant or
    ignored -- verifies the integration point recommendation_engine.py's
    plan_activities relies on."""
    from recommendation_engine import plan_activities
    from chat_assistant_prompt import new_traveller_profile

    profile = new_traveller_profile()
    profile["trip"]["destination"]["confirmed"] = ["Agra"]
    profile["interests"] = {"history": 9, "photography": 8}
    profile["trip"]["stay_location"]["nearby_points_of_interest"] = [
        "Idgah railway station", "Shahi Jama Masjid Agra", "District Hospital Agra",
    ]

    results = plan_activities(profile, llm_client=None)
    sources = {r["source"] for r in results}
    assert "curated_corpus" in sources, "Curated Agra content should still appear"
    assert "live_places_data" in sources, "Real fetched POIs should be included too"
    print("PASS: live Google Places data and curated corpus content combine into one grounded pool")


def test_destination_outside_corpus_still_works_via_live_data():
    """A destination with zero curated coverage should still produce
    grounded (if unranked) results when live Places POIs are available --
    this is a direct improvement over the pre-integration behavior, which
    returned nothing for any destination outside the small seed corpus."""
    from recommendation_engine import plan_activities
    from chat_assistant_prompt import new_traveller_profile

    profile = new_traveller_profile()
    profile["trip"]["destination"]["confirmed"] = ["Udaipur"]  # not in rag/documents.py
    profile["trip"]["stay_location"]["nearby_points_of_interest"] = [
        "City Palace Udaipur", "Lake Pichola",
    ]

    results = plan_activities(profile, llm_client=None)
    assert len(results) == 2
    assert all(r["source"] == "live_places_data" for r in results)
    print("PASS: destinations outside curated coverage still produce grounded results via live Places data")


if __name__ == "__main__":
    test_semantic_ranking_is_sensible()
    test_destination_filtering()
    test_unknown_destination_returns_empty_not_crash()
    test_hard_filter_excludes_incompatible_activities()
    test_llm_judgment_layer_drops_fabricated_ids()
    test_no_destination_degrades_gracefully()
    test_live_places_and_curated_content_combine()
    test_destination_outside_corpus_still_works_via_live_data()
    print("\nAll RAG tests passed.")
