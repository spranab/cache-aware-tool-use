from __future__ import annotations

_NAMES = [
    "Alice Chen", "Bob Smith", "Carol Davis", "Dave Johnson", "Eve Patel",
    "Frank Lopez", "Grace Kim", "Hugo Martin", "Ivy Reyes", "Jay Liu",
    "Kira Singh", "Leo Park", "Mia Russo", "Noah Bennett", "Olive Tan",
    "Priya Shah", "Quentin Hale", "Riya Iyer", "Sam Cohen", "Tara Brooks",
]
_COMPANIES = [
    "Acme Corp", "Northwind", "Globex", "Initech", "Hooli", "Stark Industries",
    "Wonka Inc", "Soylent", "Cyberdyne", "Wayne Enterprises", "Massive Dynamic",
    "Pied Piper", "Dunder Mifflin", "Aperture", "Tyrell", "Umbrella",
]
_ROLES = [
    "software engineer", "product manager", "data scientist", "designer",
    "founder", "operations lead", "marketing director", "researcher",
    "site reliability engineer", "technical writer",
]
_STYLES = ["formal", "casual", "concise", "detailed", "direct"]
_TRAITS = ["thorough", "decisive", "analytical", "creative", "pragmatic"]


def persona_text(user_id: int) -> str:
    n = _NAMES[user_id % len(_NAMES)]
    c = _COMPANIES[(user_id // 1) % len(_COMPANIES)]
    r = _ROLES[(user_id // 2) % len(_ROLES)]
    s = _STYLES[(user_id // 3) % len(_STYLES)]
    t = _TRAITS[(user_id // 5) % len(_TRAITS)]
    return (
        f"You are assisting {n} at {c}, a {r}. "
        f"They prefer {s} responses and value being {t}. "
        f"Account: user_{user_id:05d}."
    )
