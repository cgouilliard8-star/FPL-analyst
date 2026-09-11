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
PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"
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
    for element in snapshot["elements"]:
        code = element["code"]
        path = directory / f"p{code}.png"
        if path.exists() and path.stat().st_size > 0:
            present += 1
            continue
        try:
            response = getter(PHOTO_URL.format(code=code), timeout=TIMEOUT, headers=HEADERS)
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
