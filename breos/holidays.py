"""Public-holiday calendars computed for any year.

Load profiles use them to type days, and tariff schedules may name one
instead of listing their holiday dates.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Callable


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (the anonymous Gregorian computus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    return date(year, month, (h + l - 7 * m + 33 * month + 19) % 32)


# The first year of the Portuguese holiday calendar. The 2003 Labour Code
# (Article 208), in force from December 2003, is the earliest source for the
# full list below.
_PORTUGAL_CALENDAR_FIRST_YEAR = 2004


@lru_cache(maxsize=None)
def _portugal_national_holidays(year: int) -> frozenset[date]:
    """Portugal's nationwide statutory holidays in ``year``.

    These are the mandatory holidays of the Labour Code (Article 234): 1
    January, Good Friday, Easter Sunday, 25 April, 1 May, Corpus Christi
    (Easter + 60 days), 10 June, 15 August, 5 October, 1 November and 1, 8
    and 25 December. Law 23/2012 suspended Corpus Christi, 5 October,
    1 November and 1 December from 1 January 2013; Law 8/2016 restored them
    from 2 April 2016, before any of the four fell that year. They are
    therefore absent in 2013, 2014 and 2015 only.

    The calendar starts in 2004. The 2003 Labour Code (Article 208), in
    force from December 2003, gives the same list, and the 2009 Code kept
    it; this helper has no source for earlier years. Years after the
    current one are assumed to keep the current list.

    The optional holidays of Article 235 (Carnival Tuesday and the municipal
    holiday), local Good Friday substitutions, bridge days and government
    tolerances are not included: they vary by locality or employer. This
    calendar types the days of dated E-REDES BTN profiles, and tariff
    schedules that name ``pt_national`` take their holidays from it.

    Raises:
        ValueError: If ``year`` is before 2004.
    """
    if year < _PORTUGAL_CALENDAR_FIRST_YEAR:
        raise ValueError(
            f"BREOS's Portuguese national-holiday calendar starts in {_PORTUGAL_CALENDAR_FIRST_YEAR}; "
            f"it has no holiday list for {year}."
        )
    easter = _easter_sunday(year)
    days = {
        date(year, 1, 1),
        easter - timedelta(days=2),
        easter,
        date(year, 4, 25),
        date(year, 5, 1),
        date(year, 6, 10),
        date(year, 8, 15),
        date(year, 12, 8),
        date(year, 12, 25),
    }
    if not 2013 <= year <= 2015:
        days |= {easter + timedelta(days=60), date(year, 10, 5), date(year, 11, 1), date(year, 12, 1)}
    return frozenset(days)


# Calendars a tariff schedule may name in ``holidays.calendar``.
NAMED_HOLIDAY_CALENDARS: dict[str, Callable[[int], frozenset[date]]] = {
    "pt_national": _portugal_national_holidays,
}
