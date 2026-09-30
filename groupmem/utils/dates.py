"""Deterministic relative-date arithmetic for the read side.

Lifted verbatim (behavior-preserving) out of ``ReadTools.resolve_date``, kept as a
separate module because the arithmetic is the one part of the date tool with no
LLM in it.

``resolve_relative_date(base, phrase) -> datetime | None`` returns None when the
phrase is not one of the recognized patterns (the caller then keeps the original
surface text rather than guessing).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

_WEEKDAY = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}

_WD_ABBR = {"mon": "monday", "tues": "tuesday", "wednes": "wednesday",
            "thurs": "thursday", "fri": "friday", "satur": "saturday",
            "sun": "sunday"}

_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# marks a PAST-direction reading of a weekday name ("last Friday")
_PAST_CUE = re.compile(r"\b(last|previous|past|prior)\b")

# "3 days ago" / "two years ago" (digits or 1-10 as words)
_MAGO = re.compile(r"(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine|ten))\s+"
                   r"(day|week|month|year)s?\s+ago")

_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _shift_months(dt: datetime, n: int) -> datetime:
    """Shift dt by n calendar months (n may be negative), clamping the day
    when the target month is shorter (Mar 31 - 1 month -> Feb 28/29)."""
    mo = dt.month - 1 + n
    yr = dt.year + mo // 12
    mo = mo % 12 + 1
    leap = yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)
    cap = 29 if (mo == 2 and leap) else _MONTH_DAYS[mo - 1]
    return dt.replace(year=yr, month=mo, day=min(dt.day, cap))


def _shift_years(dt: datetime, n: int) -> datetime:
    try:
        return dt.replace(year=dt.year + n)
    except ValueError:  # Feb 29 anchor into a non-leap target year
        return dt.replace(year=dt.year + n, day=28)


def resolve_relative_date(base: Optional[datetime], phrase: str) -> Optional[datetime]:
    """Resolve `phrase` against the anchor datetime `base`.

    Recognized: tomorrow/today/EOD, end of week/month, next week, in N weeks,
    in N business days, in N days, weekday names (optionally 'next'/'following'
    for future or 'last'/'previous'/'past' for past), 'by the Nth', and the
    past-direction conversational set — yesterday/last night, last/next/this
    weekend, last week/month/year, next month, N days/weeks/months/years ago.
    """
    if not base or base.year <= 1:
        return None
    p = (phrase or "").lower()
    if not p:
        return None

    mday = re.search(r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b", p)
    mweeks = re.search(r"in\s+(\d+)\s+weeks?\b", p)
    mbdays = re.search(r"in\s+(\d+)\s+business\s+days?\b", p)
    mnd = re.search(r"in\s+(\d+)\s+day", p) or re.search(r"(\d+)\s+days?\b", p)
    mnth = re.search(r"by the\s+(\d+)(st|nd|rd|th)\b", p)

    # --- past-direction conversational phrasings ("yesterday", "last week",
    # "two years ago"). Must run BEFORE the future-direction branches: "5 days
    # ago" would otherwise be eaten by the N-days pattern and resolved to a
    # FUTURE date. ---
    if "yesterday" in p or "last night" in p:
        return base - timedelta(days=1)
    mago = _MAGO.search(p)
    if mago:
        n = int(mago.group(1)) if mago.group(1) else _WORDNUM[mago.group(2)]
        unit = mago.group(3)
        if unit == "day":
            return base - timedelta(days=n)
        if unit == "week":
            return base - timedelta(weeks=n)
        if unit == "month":
            return _shift_months(base, -n)
        return _shift_years(base, -n)
    if "last weekend" in p or "past weekend" in p:
        # Saturday that began the most recent FINISHED weekend
        return base - timedelta(days=(base.weekday() + 2) % 7 or 7)
    if "last week" in p or "past week" in p:
        return base - timedelta(weeks=1)
    if "this weekend" in p or "next weekend" in p:
        return base + timedelta(days=(5 - base.weekday()) % 7)  # coming Saturday
    if "last month" in p or "past month" in p:
        return _shift_months(base, -1)
    if "next month" in p:
        return _shift_months(base, 1)
    if "last year" in p or "past year" in p:
        return _shift_years(base, -1)

    if "tomorrow" in p:
        return base + timedelta(days=1)
    if "today" in p or "end of day" in p or ("eod" in p and not mday):
        return base
    if "end of the month" in p or "end of month" in p:
        if base.month == 12:
            nxt = base.replace(year=base.year + 1, month=1, day=1)
        else:
            nxt = base.replace(month=base.month + 1, day=1)
        return nxt - timedelta(days=1)
    if "end of the week" in p or "end of week" in p or "eow" in p:
        return base + timedelta(days=(4 - base.weekday()) % 7)  # coming Friday
    if ("next week" in p or "following week" in p) and not mday:
        return base + timedelta(days=(7 - base.weekday()) % 7 or 7)
    if mweeks:
        return base + timedelta(weeks=int(mweeks.group(1)))
    if mbdays:
        n, cur = int(mbdays.group(1)), base
        while n > 0:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                n -= 1
        return cur
    if mday:
        name = _WD_ABBR[mday.group(1)]
        if _PAST_CUE.search(p):
            # "last Friday" — most recent PAST occurrence of that weekday.
            # Without this the generic branch silently resolved it to the
            # coming (future) Friday, handing the agent a confident wrong date.
            delta = (base.weekday() - _WEEKDAY[name]) % 7 or 7
            return base - timedelta(days=delta)
        delta = (_WEEKDAY[name] - base.weekday()) % 7
        if delta == 0 or ("next" in p or "following" in p):
            delta = delta + 7 if ("next" in p or "following" in p) else delta
        return base + timedelta(days=delta)
    if mnth:
        day = int(mnth.group(1))
        try:
            cand = base.replace(day=day)
        except ValueError:
            cand = None
        if cand is None or cand.date() < base.date():
            mo = base.month + 1
            yr = base.year + (1 if mo > 12 else 0)
            mo = 1 if mo > 12 else mo
            try:
                cand = base.replace(year=yr, month=mo, day=day)
            except ValueError:
                cand = None
        return cand
    if mnd:
        return base + timedelta(days=int(mnd.group(1)))
    return None
