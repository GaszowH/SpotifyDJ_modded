"""
cli.py
------
Headless CLI mode for Spotify AI DJ.

Invoked automatically by main.py when arguments are passed:
  dj "play some dark techno"
  python main.py "relaxing lo-fi"

Prints coloured status output to the terminal and exits when done.
No GUI is launched. Safe to run over SSH or in scripts.
"""

import sys
import re as _re

from brain import get_vibe_params, get_playlist_vibe_params, get_continue_params
from config import is_configured, load_config
from spotify_client import SpotifyClient

# ------------------------------------------------------------------
# Colour helpers (auto-disabled when output is not a terminal)
# ------------------------------------------------------------------
if sys.stdout.isatty():
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _CYAN   = "\033[36m"
else:
    _RESET = _BOLD = _GREEN = _YELLOW = _RED = _CYAN = ""

def _info(msg: str)    -> None: print(f"{_CYAN}{_BOLD}[*]{_RESET} {msg}")
def _success(msg: str) -> None: print(f"{_GREEN}{_BOLD}[+]{_RESET} {msg}")
def _warn(msg: str)    -> None: print(f"{_YELLOW}{_BOLD}[!]{_RESET} {msg}")
def _error(msg: str)   -> None: print(f"{_RED}{_BOLD}[x]{_RESET} {msg}")

# Persists between CLI calls within the same process
_cli_spotify_client: SpotifyClient | None = None

def _get_cli_client() -> SpotifyClient:
    global _cli_spotify_client
    if _cli_spotify_client is None:
        _cli_spotify_client = SpotifyClient()
    return _cli_spotify_client


def run_cli(request: str, is_continue: bool = False) -> int:
    if not is_configured():
        _error("No Gemini API key found.")
        _warn(
            "Run the app in GUI mode first to complete setup:\n"
            "  python main.py\n"
            "Or set your key directly:\n"
            "  python main.py --set-key YOUR_KEY_HERE"
        )
        return 1

    config     = load_config()
    api_key    = config.get("gemini_api_key", "")
    local_only = config.get("local_ai_only", False)
    client     = _get_cli_client()

    playlist_url = _re.search(
        r"(https?://open\.spotify\.com/playlist/[A-Za-z0-9]+|spotify:playlist:[A-Za-z0-9]+)",
        request or ""
    )

    # STEP 0: Fetch user's top artists/genres for playlist context
    user_taste = client.get_user_top_artists_and_genres()
    playlist_context = user_taste.get("artists", []) + user_taste.get("genres", [])

    # STEP 1: AI generates search queries
    if is_continue:
        if not client.last_request:
            _error("Nothing playing yet — run a normal request first.")
            return 1
        _info(f'Continuing: "{client.last_request}"')
        try:
            directives = get_continue_params(
                client.last_request, 
                client.last_queries, 
                api_key, 
                local_only=local_only,
                playlist_context=playlist_context
            )
        except Exception as e:
            _error(f"AI error: {e}")
            return 1
        playlist_tracks = None

    elif playlist_url:
        _info("Playlist URL detected — fetching tracks...")
        try:
            playlist_tracks = client.get_playlist_tracks(playlist_url.group(0))
            _info(f"Fetched {len(playlist_tracks)} tracks from playlist")
        except Exception as e:
            _error(f"Playlist error: {e}")
            return 1
        user_intent = _re.sub(r"https?://\S+|spotify:\S+", "", request).strip()
        client.last_request = request
        try:
            directives = get_playlist_vibe_params(
                playlist_tracks, user_intent, api_key, local_only=local_only,
                playlist_context=playlist_context
            )
        except Exception as e:
            _error(f"AI error: {e}")
            return 1

    else:
        playlist_tracks = None
        _info(f'Request: "{request}"')
        client.last_request = request
        try:
            directives = get_vibe_params(
                request, api_key, local_only=local_only,
                playlist_context=playlist_context
            )
        except Exception as e:
            _error(f"AI error: {e}")
            return 1

    _info(f"AI: {directives.reasoning}")
    _info(f"Queries ({len(directives.queries)}): {directives.queries}")
    _info(f"Target queue: {directives.queue_size} tracks")

    # STEP 1.5: Process seed_artists if provided by the AI
    extra_tracks = []
    if hasattr(directives, "seed_artists") and directives.seed_artists:
        _info(f"AI provided seed artists: {directives.seed_artists}")
        _info("Fetching related artists from Spotify's graph...")
        extra_tracks = client.get_related_artist_tracks(directives.seed_artists, max_tracks=40)

    # STEP 2: Search Spotify and start playback
    try:
        if playlist_tracks is not None:
            result = client.search_and_play_mixed(playlist_tracks, directives, extra_tracks=extra_tracks)
        else:
            result = client.search_and_play(directives, extra_tracks=extra_tracks)
    except Exception as e:
        _error(f"Spotify error: {e}")
        return 1

    if result.success:
        _success(f"Now playing: {_BOLD}{result.first_track}{_RESET}")
        _info(f"{result.track_count} tracks queued from {result.queries_run} searches")
        return 0
    else:
        _error(result.message)
        return 1

def run_set_key(key: str) -> int:
    from config import save_config
    key = key.strip()
    if len(key) < 20:
        _error("That doesn't look like a valid key.")
        return 1
    config = load_config()
    config["gemini_api_key"] = key
    save_config(config)
    _success("Gemini API key saved.")
    _info('You can now use the CLI: dj "your request"')
    return 0

def print_help() -> None:
    print(f"""\
{_BOLD}Spotify AI DJ{_RESET}

{_BOLD}GUI mode{_RESET} (no arguments):
  python main.py
  dj

{_BOLD}CLI mode{_RESET} (play immediately from terminal):
  dj "dark techno"
  python main.py "90s hip hop"

{_BOLD}Continue playing{_RESET} (fresh tracks, same vibe):
  dj --continue

{_BOLD}First-time setup from terminal{_RESET}:
  python main.py --set-key YOUR_GEMINI_API_KEY
""")
