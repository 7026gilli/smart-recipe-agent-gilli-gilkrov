"""Isolated offline unit tests, one class per module in API/harness/.
No API key or network. Reuses the mock-model fixtures from test_offline.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace
from typing import Any

from tests.test_offline import OfflineTestCase, make_response

with contextlib.redirect_stdout(io.StringIO()):
    import shared
    from google.genai import types
    from API.harness import caching, compaction, context, loop, memory, tokens, tools
    from API.harness.caching import PromptCache
    from API.harness.history import History
    from API.harness.tokens import TokenTracker


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"function_call": {"name": name, "args": args}}


class LoopUnitTests(OfflineTestCase):
    def _run(self, responses: list[Any], **kw: Any) -> tuple[dict[str, Any], History, list[int]]:
        h, t, seen = History(), TokenTracker(), []
        h.append_user("hi")
        it = iter(responses)

        def send(hist: History) -> Any:
            seen.append(len(hist))
            return next(it)

        return loop.run_turn(h, t, send, **kw), h, seen

    def test_terminates_on_end_turn_after_multiple_tool_rounds(self) -> None:
        res, h, seen = self._run([
            make_response([_call("search_recipes", {"query": "pasta"})]),
            make_response([_call("fetch_external_nutrition_info", {"ingredient_or_dish": "pasta"})]),
            make_response([{"text": "All done"}]),
        ])
        self.assertEqual(res["stop_reason"], "end_turn")
        self.assertEqual(res["text"], "All done")
        self.assertEqual(len(seen), 3)
        self.assertEqual([c["name"] for c in res["tool_calls"]],
                         ["search_recipes", "fetch_external_nutrition_info"])

    def test_parallel_calls_answered_in_one_user_message(self) -> None:
        res, h, _ = self._run([
            make_response([_call("search_recipes", {"query": "soup"}),
                           _call("fetch_external_nutrition_info", {"ingredient_or_dish": "rice"})]),
            make_response([{"text": "ok"}]),
        ])
        responses_msg = h.messages[2]
        self.assertEqual(responses_msg["role"], "user")
        self.assertEqual(len(responses_msg["parts"]), 2)
        self.assertTrue(all("function_response" in p for p in responses_msg["parts"]))
        self.assertEqual([m["role"] for m in h.messages], ["user", "model", "user", "model"])

    def test_stops_at_max_iterations(self) -> None:
        res, h, seen = self._run([make_response([_call("search_recipes", {"query": "x"})])] * 3,
                                 max_iterations=3)
        self.assertEqual(res["stop_reason"], "max_iterations")
        self.assertEqual(len(seen), 3)

    def test_stop_reason_mapping(self) -> None:
        r = make_response([{"text": "x"}], finish="MAX_TOKENS")
        self.assertEqual(loop.stop_reason(r, loop.response_parts(r)), "max_tokens")
        r = make_response([_call("search_recipes", {})])
        self.assertEqual(loop.stop_reason(r, loop.response_parts(r)), "tool_use")


class ToolsUnitTests(OfflineTestCase):
    def test_schema_structure(self) -> None:
        self.assertEqual({d["name"] for d in tools.TOOL_DECLARATIONS},
                         {"search_recipes", "save_user_preference_or_recipe",
                          "fetch_external_nutrition_info"})
        for d in tools.TOOL_DECLARATIONS:
            self.assertTrue(d["description"])
            schema = d["parameters_json_schema"]
            self.assertEqual(schema["type"], "object")
            self.assertTrue(set(schema["required"]) <= set(schema["properties"]))
        built = tools.get_tools()
        self.assertEqual(len(built[0].function_declarations), 3)

    def test_dispatch_success(self) -> None:
        part, is_error = tools.execute_tool_call(
            {"name": "search_recipes", "args": {"query": "pasta"}, "id": "c1"})
        self.assertFalse(is_error)
        fr = part["function_response"]
        self.assertEqual((fr["name"], fr["id"]), ("search_recipes", "c1"))
        self.assertIn("result", fr["response"])

    def test_unknown_tool_returns_error_dict(self) -> None:
        part, is_error = tools.execute_tool_call({"name": "nope", "args": {}})
        self.assertTrue(is_error)
        self.assertIn("Unknown tool", part["function_response"]["response"]["error"])

    def test_invalid_args_type_returns_error_dict(self) -> None:
        part, is_error = tools.execute_tool_call({"name": "search_recipes", "args": "bad"})
        self.assertTrue(is_error)
        self.assertIn("error", part["function_response"]["response"])

    def test_handler_exception_is_caught(self) -> None:
        def boom(_: dict[str, Any]) -> Any:
            raise ValueError("kaput")
        with unittest_patch_handler("search_recipes", boom):
            part, is_error = tools.execute_tool_call({"name": "search_recipes", "args": {"query": "x"}})
        self.assertTrue(is_error)
        self.assertEqual(part["function_response"]["response"], {"error": "ValueError: kaput"})

    def test_ok_false_result_becomes_error(self) -> None:
        part, is_error = tools.execute_tool_call(
            {"name": "save_user_preference_or_recipe", "args": {"type": "preference"}})
        self.assertTrue(is_error)
        self.assertIn("error", part["function_response"]["response"])

    def test_execute_tool_calls_batches(self) -> None:
        parts, errs = tools.execute_tool_calls([
            {"name": "search_recipes", "args": {"query": "pasta"}},
            {"name": "nope", "args": {}}])
        self.assertEqual(len(parts), 2)
        self.assertEqual(errs, [False, True])


@contextlib.contextmanager
def unittest_patch_handler(name: str, fn: Any):
    old = tools.HANDLERS[name]
    tools.HANDLERS[name] = fn
    try:
        yield
    finally:
        tools.HANDLERS[name] = old


class HistoryUnitTests(OfflineTestCase):
    def test_append_only_and_role_merge(self) -> None:
        h = History()
        h.append_user("a")
        h.append_user("b")
        h.append_model([{"text": "m"}])
        self.assertEqual(len(h), 2)
        self.assertEqual(h.messages[0]["parts"], [{"text": "a"}, {"text": "b"}])

    def test_persist_and_reload_across_sessions(self) -> None:
        path = self.tmpdir / "sub" / "hist.json"
        h = History()
        h.append_user("hello")
        h.append_model([{"text": "hi"}])
        h.save(path)
        h.append_user("later")
        h.save(path)  # a later save keeps everything written before
        loaded = History.load(path)
        self.assertEqual(loaded.messages, h.messages)
        loaded.append_model([{"text": "again"}])
        self.assertEqual(len(loaded), 4)
        self.assertEqual(len(History.load(path)), 3)  # in-memory change not persisted until save

    def test_load_missing_file_is_empty(self) -> None:
        self.assertEqual(len(History.load(self.tmpdir / "none.json")), 0)

    def test_snapshot_is_independent_copy(self) -> None:
        h = History()
        h.append_user("a")
        snap = h.snapshot()
        snap[0]["parts"].append({"text": "x"})
        self.assertEqual(len(h.messages[0]["parts"]), 1)

    def test_compactions_counter_roundtrip(self) -> None:
        path = self.tmpdir / "h.json"
        h = History()
        h.replace_all([])
        h.save(path)
        self.assertEqual(History.load(path).compactions, 1)


class MemoryUnitTests(OfflineTestCase):
    def test_save_and_retrieve_preferences(self) -> None:
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegetarian"})
        facts = memory.relevant_facts("anything at all")
        self.assertIn("diet: vegetarian", facts)

    def test_update_preference_overwrites(self) -> None:
        for v in ("vegetarian", "vegan"):
            shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": v})
        facts = memory.relevant_facts("hi")
        self.assertIn("diet: vegan", facts)
        self.assertNotIn("diet: vegetarian", facts)

    def test_contradicting_request_keeps_stored_preference(self) -> None:
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegetarian"})
        block = memory.format_memory_block("Give me the Chicken Stir Fry recipe")
        self.assertIn("diet: vegetarian", block)

    def test_relevant_notes_preferred_over_recent(self) -> None:
        mem = {"preferences": {}, "notes": [{"text": "likes basil"}, {"text": "dislikes cilantro"}],
               "saved_recipes": []}
        self.assertEqual(memory.relevant_facts("what about cilantro", mem), ["note: dislikes cilantro"])

    def test_saved_recipe_titles_listed(self) -> None:
        mem = {"preferences": {}, "notes": [],
               "saved_recipes": [{"id": "r5", "title": "Tomato Basil Pasta"}]}
        facts = memory.relevant_facts("pasta please", mem)
        self.assertEqual(facts, ["saved recipes: Tomato Basil Pasta (id r5)"])

    def test_empty_memory_gives_no_block(self) -> None:
        self.assertIsNone(memory.format_memory_block("hello"))


class ContextUnitTests(OfflineTestCase):
    def test_irrelevant_notes_excluded(self) -> None:
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegetarian"})
        shared.save_user_preference_or_recipe({"type": "note", "text": "hates cilantro"})
        shared.save_user_preference_or_recipe({"type": "note", "text": "owns a pressure cooker"})
        parts = context.build_user_turn("tell me about cilantro")
        text = "\n".join(p["text"] for p in parts)
        self.assertIn("hates cilantro", text)
        self.assertNotIn("pressure cooker", text)
        self.assertIn("diet: vegetarian", text)  # preferences always included

    def test_part_order_summary_memory_text(self) -> None:
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegan"})
        parts = context.build_user_turn("hello", summary="earlier stuff")
        self.assertTrue(parts[0]["text"].startswith("<conversation_summary>"))
        self.assertTrue(parts[1]["text"].startswith("<user_memory>"))
        self.assertEqual(parts[2], {"text": "hello"})

    def test_no_memory_no_summary_is_just_text(self) -> None:
        self.assertEqual(context.build_user_turn("hello"), [{"text": "hello"}])

    def test_config_disables_sdk_afc_and_toggles_tools(self) -> None:
        cache = PromptCache(None, "m", context.SYSTEM_PROMPT, tools.get_tools())
        cfg = context.build_config(cache, max_output_tokens=100, thinking_budget=0)
        self.assertTrue(cfg.automatic_function_calling.disable)
        self.assertEqual(cfg.tool_config.function_calling_config.mode.value, "AUTO")
        cfg = context.build_config(cache, max_output_tokens=100, thinking_budget=None, allow_tools=False)
        self.assertEqual(cfg.tool_config.function_calling_config.mode.value, "NONE")
        self.assertIsNone(cfg.thinking_config)


class CompactionUnitTests(OfflineTestCase):
    def _history(self, n: int = 6, size: int = 400) -> History:
        h = History()
        for i in range(n):
            h.append_user("u" * size + str(i))
            h.append_model([{"text": "m" * size}])
        return h

    def test_triggers_above_threshold_using_estimate(self) -> None:
        h, t = self._history(), TokenTracker()
        self.assertTrue(compaction.needs_compaction(h, t, threshold=100))
        self.assertFalse(compaction.needs_compaction(h, t, threshold=10_000))

    def test_uses_reported_usage_when_available(self) -> None:
        h, t = self._history(), TokenTracker()
        t.record({"prompt_token_count": 5000})
        self.assertTrue(compaction.needs_compaction(h, t, threshold=4000))
        self.assertFalse(compaction.needs_compaction(h, t, threshold=6000))

    def test_tiny_history_never_compacts(self) -> None:
        h = History()
        h.append_user("x" * 10_000)
        self.assertFalse(compaction.needs_compaction(h, TokenTracker(), threshold=1))

    def test_compact_clears_history_and_returns_summary(self) -> None:
        h = self._history()
        seen: list[History] = []

        def summarize(req: History) -> str:
            seen.append(req)
            return "SUMMARY"

        self.assertEqual(compaction.compact(h, summarize), "SUMMARY")
        self.assertEqual(len(h), 0)
        self.assertEqual(h.compactions, 1)
        self.assertIn(compaction.SUMMARY_PROMPT, json.dumps(seen[0].messages))

    def test_fallback_when_summarizer_raises_or_empty(self) -> None:
        for summarizer in (lambda r: (_ for _ in ()).throw(RuntimeError("down")), lambda r: None):
            h = self._history(2, 20)
            summary = compaction.compact(h, summarizer)
            self.assertIn("user:", summary)
            self.assertLessEqual(len(summary), compaction.MAX_FALLBACK_CHARS)


class CachingUnitTests(OfflineTestCase):
    def test_prefix_identical_across_calls(self) -> None:
        cache = PromptCache(None, "m", context.SYSTEM_PROMPT, tools.get_tools())
        a, b = cache.prefix_config(), cache.prefix_config()
        self.assertEqual(a["system_instruction"], b["system_instruction"])
        self.assertIs(a["tools"], b["tools"])
        self.assertNotIn("cached_content", a)

    def test_tool_declarations_deterministic(self) -> None:
        dump = lambda: json.dumps(tools.get_tools()[0].model_dump(mode="json", exclude_none=True))
        self.assertEqual(dump(), dump())

    def test_explicit_cache_used_when_prefix_large(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(count_tokens=lambda **k: SimpleNamespace(total_tokens=5000)),
            caches=SimpleNamespace(create=lambda **k: SimpleNamespace(name="cachedContents/1")))
        cache = PromptCache(client, "m", "sys", [], explicit=True)
        self.assertEqual(cache.prefix_config(), {"cached_content": "cachedContents/1"})

    def test_fallback_when_prefix_small_or_api_fails(self) -> None:
        small = SimpleNamespace(models=SimpleNamespace(count_tokens=lambda **k: SimpleNamespace(total_tokens=10)))
        self.assertIsNone(PromptCache(small, "m", "sys", [], explicit=True).cache_name)

        def fail(**k: Any) -> Any:
            raise RuntimeError("quota")
        broken = SimpleNamespace(models=SimpleNamespace(count_tokens=fail))
        cache = PromptCache(broken, "m", "sys", [], explicit=True)
        self.assertIsNone(cache.cache_name)
        self.assertIn("system_instruction", cache.prefix_config())

    def test_log_hit_handles_missing_fields(self) -> None:
        PromptCache.log_hit(SimpleNamespace(prompt_token_count=None, cached_content_token_count=None))
        PromptCache.log_hit(SimpleNamespace(prompt_token_count=100, cached_content_token_count=40))


class TokensUnitTests(OfflineTestCase):
    def test_record_separates_input_output_thinking_cached(self) -> None:
        t = TokenTracker()
        entry = t.record({"prompt_token_count": 1000, "cached_content_token_count": 400,
                          "candidates_token_count": 50, "thoughts_token_count": 30})
        self.assertEqual(entry, {"input": 600, "output": 80, "thinking": 30,
                                 "cache_read": 400, "cache_write": 0})
        t.record(types.GenerateContentResponseUsageMetadata(
            prompt_token_count=200, candidates_token_count=10))
        self.assertEqual((t.input_tokens, t.output_tokens, t.thinking_tokens, t.cache_read_tokens),
                         (800, 90, 30, 400))
        self.assertEqual((t.requests, t.last_context_tokens), (2, 200))
        self.assertEqual(len(t.per_request), 2)

    def test_tool_use_prompt_tokens_count_as_input(self) -> None:
        t = TokenTracker()
        t.record({"prompt_token_count": 100, "tool_use_prompt_token_count": 25})
        self.assertEqual(t.input_tokens, 125)

    def test_free_tier_cost_is_zero(self) -> None:
        t = TokenTracker()
        t.record({"prompt_token_count": 10_000, "candidates_token_count": 5_000})
        self.assertEqual(t.cost_usd(), 0.0)
        self.assertEqual(t.summary()["estimated_cost_usd"], 0.0)

    def test_paid_pricing_formula(self) -> None:
        t = TokenTracker(model="m", tier="paid")
        t.record({"prompt_token_count": 1_000_000, "cached_content_token_count": 500_000,
                  "candidates_token_count": 1_000_000})
        with unittest_patch_pricing({"paid": {"m": (1.0, 4.0, 0.0, 0.25)}}):
            # 500k uncached in @1 + 1M out @4 + 500k cached @0.25 = 0.5 + 4 + 0.125
            self.assertAlmostEqual(t.cost_usd(), 4.625)

    def test_over_budget_and_estimate(self) -> None:
        t = TokenTracker()
        t.record({"prompt_token_count": 500})
        self.assertTrue(t.over_budget(499))
        self.assertFalse(t.over_budget(500))
        self.assertEqual(tokens.estimate_tokens("a" * 40), 10)
        self.assertEqual(tokens.estimate_tokens(""), 1)

    def test_summary_keys(self) -> None:
        s = TokenTracker().summary()
        self.assertEqual(set(s), {"model", "requests", "input_tokens", "output_tokens", "thinking_tokens",
                                  "cache_read_tokens", "cache_write_tokens", "estimated_cost_usd"})


@contextlib.contextmanager
def unittest_patch_pricing(table: dict[str, Any]):
    old = dict(tokens.PRICING)
    tokens.PRICING.update(table)
    try:
        yield
    finally:
        tokens.PRICING.clear()
        tokens.PRICING.update(old)
