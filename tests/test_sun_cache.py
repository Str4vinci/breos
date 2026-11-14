"""The kept sun positions and clear-sky frames: what may and may not be reused."""

import pandas as pd
import pytest
from pvlib.location import Location

from breos import _sun_cache

SITE = Location(41.15, -8.61, altitude=100.0)


def _times(tz=None):
    times = pd.date_range("2025-06-01 06:00", periods=6, freq="h")
    return times if tz is None else times.tz_localize(tz)


@pytest.mark.parametrize("kind", ["solar_position", "clear_sky"])
def test_kept_frame_comes_back_on_the_clock_asked_for(kind):
    method = SITE.get_solarposition if kind == "solar_position" else SITE.get_clearsky
    naive, aware = _times(), _times("UTC")
    _sun_cache.sun_frame(kind, SITE, naive, lambda: method(naive))
    kept = _sun_cache.sun_frame(kind, SITE, aware, lambda: method(aware))
    assert kept.index.equals(aware) and kept.index.tz == aware.tz
    pd.testing.assert_frame_equal(kept, method(aware))


def test_clear_sky_is_not_reused_once_the_sites_sun_is_replaced(monkeypatch):
    times = _times("UTC")
    _sun_cache.sun_frame("clear_sky", SITE, times, lambda: SITE.get_clearsky(times))
    original = Location.get_solarposition

    def shifted(self, *args, **kwargs):
        position = original(self, *args, **kwargs)
        position["apparent_zenith"] += 10.0
        return position

    monkeypatch.setattr(Location, "get_solarposition", shifted)
    fresh = _sun_cache.sun_frame("clear_sky", SITE, times, lambda: SITE.get_clearsky(times))
    pd.testing.assert_frame_equal(fresh, SITE.get_clearsky(times))


def test_returned_frames_are_copies():
    times = _times("UTC")
    first = _sun_cache.sun_frame("solar_position", SITE, times, lambda: SITE.get_solarposition(times))
    first.iloc[:, :] = -1.0
    again = _sun_cache.sun_frame("solar_position", SITE, times, lambda: SITE.get_solarposition(times))
    pd.testing.assert_frame_equal(again, SITE.get_solarposition(times))


def test_renaming_a_returned_index_touches_neither_the_request_nor_the_kept_frame():
    times = _times("UTC")
    first = _sun_cache.sun_frame("solar_position", SITE, times, lambda: SITE.get_solarposition(times))
    first.index.name = "renamed"
    assert times.name is None
    again = _sun_cache.sun_frame("solar_position", SITE, times, lambda: SITE.get_solarposition(times))
    assert again.index.name is None


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_a_worker_forked_while_the_lock_is_held_can_use_the_cache():
    import multiprocessing

    context = multiprocessing.get_context("fork")
    with _sun_cache._LOCK:
        worker = context.Process(target=_sun_cache.clear)
        worker.start()
    worker.join(timeout=30)
    if worker.is_alive():
        worker.kill()
        worker.join()
    assert worker.exitcode == 0
