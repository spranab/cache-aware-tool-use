from __future__ import annotations

import random

from bench.common import ToolDef

_VERBS = [
    "get", "list", "create", "update", "delete", "search", "send", "schedule",
    "analyze", "summarize", "convert", "fetch", "upload", "download", "publish",
    "archive", "rename", "duplicate", "validate", "encrypt", "decrypt", "compress",
    "rollback", "approve", "reject", "assign",
]

_OBJECTS = [
    "document", "record", "item", "entry", "post", "comment", "message",
    "notification", "task", "project", "user", "account", "transaction", "invoice",
    "report", "image", "video", "log", "session", "branch", "commit", "repository",
    "container", "deployment", "secret", "metric", "alert", "ticket", "lead",
    "contract", "subscription", "campaign", "experiment", "dataset", "model",
    "annotation", "workspace", "channel", "thread", "permission",
]

_DOMAINS = [
    "crm", "billing", "monitoring", "devops", "hr", "support", "marketing",
    "analytics", "auth", "storage", "compute", "messaging", "search", "scheduler",
    "audit",
]

_FIELDS = ["id", "name", "key", "uuid", "slug", "ref"]


def _make_distractor(idx: int) -> ToolDef:
    rng = random.Random(idx)
    verb = rng.choice(_VERBS)
    obj = rng.choice(_OBJECTS)
    dom = rng.choice(_DOMAINS)
    name = f"{dom}_{verb}_{obj}"
    field = rng.choice(_FIELDS)
    desc = f"{verb.capitalize()} a {obj} in the {dom} subsystem."
    return ToolDef(
        name=name,
        description=desc,
        input_schema={
            "type": "object",
            "properties": {field: {"type": "string"}},
            "required": [field],
        },
    )


def inflate_tools(existing: list[ToolDef], target_n: int, seed: int = 0) -> list[ToolDef]:
    """Pad existing tools with deterministic distractors until total == target_n.

    Real tools are preserved; distractors avoid name collisions.
    Final order is shuffled (seeded) so the real tool isn't always first.
    Returns at most target_n tools; if existing already exceeds, returns existing.
    """
    if target_n <= len(existing):
        return list(existing)

    seen = {t.name for t in existing}
    distractors: list[ToolDef] = []
    i = 0
    cap = target_n * 50
    while len(existing) + len(distractors) < target_n and i < cap:
        d = _make_distractor(seed * 10_000_000 + i)
        if d.name not in seen:
            distractors.append(d)
            seen.add(d.name)
        i += 1

    out = list(existing) + distractors
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
