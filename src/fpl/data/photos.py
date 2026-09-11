"""Player headshots for the pitch cards.

FPL publishes a cut-out portrait for every registered player at
``resources.premierleague.com/premierleague/photos/players/110x140/p<code>.png``.
The page could hot-link them, but a picture that has to come from a third party's
server is a picture that sometimes does not arrive, so the refresh copies every
current player's photo into ``site/photos/`` once (only the missing ones each run)
and the site serves them from the same place as the page. A player without a photo
falls back to FPL's URL, then to his initials.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from fpl.config import PROJECT_ROOT

log = logging.getLogger(__name__)

PHOTOS_DIR = PROJECT_ROOT / "site" / "photos"
# The image server keeps one folder per season since 2025-26 (``premierleague25``,
# ``premierleague26``...) with the current kits, and the old un-numbered folder,
# which stops being updated. Newest first; the first folder that answers for a
# known player is the season's source, and when it changes every photo is refetched.
PHOTO_SOURCES = (
    "https://resources.premierleague.com/premierleague27/photos/players/250x250/{code}.png",
    "https://resources.premierleague.com/premierleague26/photos/players/250x250/{code}.png",
    "https://resources.premierleague.com/premierleague25/photos/players/250x250/{code}.png",
    "https://resources.premierleague.com/premierleague/photos/players/250x250/p{code}.png",
    "https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png",
)
PHOTO_URL = PHOTO_SOURCES[-1]
# FPL's own silhouette for a player it has no portrait of (young signings, mostly)
MISSING_URL = (
    "https://resources.premierleague.com/premierleague/photos/players/110x140/Photo-Missing.png"
)
SOURCE_MARKER = ".source"
TIMEOUT = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (fpl-analyst; +https://github.com/cgouilliard8-star/FPL-analyst)"
}


def fetch_photos(
    snapshot: dict, *, directory: Path = PHOTOS_DIR, getter=requests.get, pause: float = 0.05
) -> dict[str, int]:
    """Download the photo of every player in the snapshot that is not already stored.

    Returns counts: fetched, already present, failed. Failures are logged and do not
    stop the run; the page has fallbacks.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fetched = present = failed = 0
    template = _pick_source(snapshot, getter)
    marker = directory / SOURCE_MARKER
    if marker.exists() and marker.read_text().strip() != template:
        log.info("photos: new season folder on the image server; refetching every portrait")
        for old in directory.glob("p*.png"):
            old.unlink()
    marker.write_text(template)
    wanted = [(e["code"], directory / f"p{e['code']}.png") for e in snapshot["elements"]]
    wanted.append(("missing", directory / "missing.png"))
    for code, path in wanted:
        if path.exists() and path.stat().st_size > 0:
            present += 1
            continue
        try:
            url = MISSING_URL if code == "missing" else template.format(code=code)
            response = getter(url, timeout=TIMEOUT, headers=HEADERS)
            if response.status_code != 200 or not response.content:
                raise RuntimeError(f"HTTP {response.status_code}")
            path.write_bytes(response.content)
            fetched += 1
            time.sleep(pause)
        except Exception as exc:  # noqa: BLE001 - a missing portrait is not an error
            failed += 1
            log.debug("photo %s: %s", code, exc)
    log.info("photos: %d fetched, %d already present, %d failed", fetched, present, failed)
    return {"fetched": fetched, "present": present, "failed": failed}


def _pick_source(snapshot: dict, getter) -> str:
    """The newest season folder that serves a portrait of a well-known player."""
    probes = sorted(snapshot["elements"], key=lambda e: -float(e.get("selected_by_percent") or 0))
    for template in PHOTO_SOURCES:
        for element in probes[:3]:
            try:
                response = getter(
                    template.format(code=element["code"]), timeout=TIMEOUT, headers=HEADERS
                )
                if response.status_code == 200 and response.content:
                    log.info("photos: using %s", template)
                    return template
            except Exception as exc:  # noqa: BLE001 - try the next folder
                log.debug("photo source %s: %s", template, exc)
    return PHOTO_URL
