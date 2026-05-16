from __future__ import annotations

from bench.common import TestCase, ToolDef


# Shared distractor pool — appears in multiple cases to test that the right tool
# is picked from a crowded set.
DISTRACTORS = [
    ToolDef(
        "send_email",
        "Send an email to a recipient.",
        {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
    ),
    ToolDef(
        "create_calendar_event",
        "Create a calendar event with a title and start time.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
            },
            "required": ["title", "start"],
        },
    ),
    ToolDef(
        "translate_text",
        "Translate text from one language to another.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    ),
    ToolDef(
        "search_news",
        "Search recent news articles.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolDef(
        "get_stock_price",
        "Get the current trading price for a stock ticker.",
        {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    ),
    ToolDef(
        "schedule_reminder",
        "Schedule a reminder for the user at a given time.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "when": {"type": "string"},
            },
            "required": ["text", "when"],
        },
    ),
]


_WEATHER_TOOL = ToolDef(
    "get_weather",
    "Get the current weather for a city.",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["metric", "imperial"]},
        },
        "required": ["city"],
    },
)


SEED_CASES: list[TestCase] = [
    TestCase(
        case_id="seed_01_weather",
        user_goal="What's the weather in Tokyo today?",
        tools=[_WEATHER_TOOL] + DISTRACTORS[:4],
        expected_tool="get_weather",
        expected_args={"city": "Tokyo"},
    ),
    TestCase(
        case_id="seed_02_email",
        user_goal="Email alice@example.com saying I'll be late to the meeting.",
        tools=DISTRACTORS,
        expected_tool="send_email",
        # Body wording will vary; only score the addressing field.
        expected_args={"to": "alice@example.com"},
    ),
    TestCase(
        case_id="seed_03_stock",
        user_goal="What's NVDA trading at right now?",
        tools=DISTRACTORS,
        expected_tool="get_stock_price",
        expected_args={"ticker": "NVDA"},
    ),
    TestCase(
        case_id="seed_04_translate",
        user_goal="Translate 'good morning' to Japanese.",
        tools=DISTRACTORS + [_WEATHER_TOOL],
        expected_tool="translate_text",
        expected_args={"target_lang": "Japanese"},
    ),
    TestCase(
        case_id="seed_05_calendar",
        user_goal="Schedule a meeting titled 'Q3 review' starting at 2025-06-10T14:00:00Z.",
        tools=DISTRACTORS,
        expected_tool="create_calendar_event",
        expected_args={"title": "Q3 review"},
    ),
]
