"""
Eval runner.

Runs golden conversations (see golden_conversations.py) against either a
MockLLMClient (deterministic harness testing, no API key needed) or a real
AnthropicLLMClient (actual model quality testing, needs ANTHROPIC_API_KEY).

Usage:
    python evals/run_evals.py                 # mock mode (default)
    python evals/run_evals.py --client live    # live mode, needs ANTHROPIC_API_KEY
    python evals/run_evals.py --scenario no_fabrication_of_family_names
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_conversations import ALL_SCENARIOS, GoldenConversation
from llm_client import MockLLMClient
from orchestrator import ConversationState, process_turn


@dataclass
class CheckResult:
    description: str
    passed: bool
    error: str | None = None


@dataclass
class ScenarioResult:
    name: str
    check_results: list[CheckResult] = field(default_factory=list)
    turn_error: str | None = None

    @property
    def passed(self) -> bool:
        if self.turn_error:
            return False
        return all(c.passed for c in self.check_results)


def run_scenario(scenario: GoldenConversation, mode: str) -> ScenarioResult:
    state = ConversationState()
    result = ScenarioResult(name=scenario.name)

    if scenario.setup:
        try:
            scenario.setup(state)
        except Exception as e:
            result.turn_error = f"setup() raised: {e}"
            return result

    if mode == "mock":
        scripted = [t.mock_response for t in scenario.turns]
        client = MockLLMClient(scripted_turns=scripted)
    else:
        from llm_client import AnthropicLLMClient
        client = AnthropicLLMClient()

    replies: list[str] = []
    try:
        for turn in scenario.turns:
            turn_result = process_turn(state, turn.user_message, client)
            replies.append(turn_result["reply"])
    except Exception as e:
        result.turn_error = f"process_turn raised: {e}"
        return result

    for check in scenario.checks:
        try:
            passed = check.fn(state.profile, replies, state)
            result.check_results.append(CheckResult(check.description, passed))
        except Exception as e:
            result.check_results.append(CheckResult(check.description, False, error=str(e)))

    return result


def print_result(scenario: GoldenConversation, result: ScenarioResult) -> None:
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"\n{status}  {scenario.name}")
    print(f"       {scenario.description}")
    if result.turn_error:
        print(f"       ⚠️  {result.turn_error}")
        return
    for check in result.check_results:
        mark = "  ✓" if check.passed else "  ✗"
        line = f"{mark} {check.description}"
        if check.error:
            line += f"  (error: {check.error})"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Run ANITA golden conversation evals")
    parser.add_argument("--client", choices=["mock", "live"], default="mock",
                         help="mock: deterministic harness test, no API key. live: real model, needs ANTHROPIC_API_KEY")
    parser.add_argument("--scenario", default=None, help="Run only the scenario with this name")
    args = parser.parse_args()

    if args.client == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: --client live requires ANTHROPIC_API_KEY to be set.")
        sys.exit(1)

    scenarios = ALL_SCENARIOS
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"ERROR: no scenario named '{args.scenario}'. Available: {[s.name for s in ALL_SCENARIOS]}")
            sys.exit(1)

    print(f"Running {len(scenarios)} scenario(s) in '{args.client}' mode...")

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, args.client)
        print_result(scenario, result)
        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 50}")
    print(f"{passed}/{total} scenarios passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
