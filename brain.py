"""
brain.py
--------
Converts a natural language music request into multiple targeted Spotify
search queries, with a strong emphasis on deep discovery, lateral moves 
based on playlist history, and finding artists outside the LLM's memory.

AI fallback chain (tried in order):
  1. Google Gemini (cloud) — fast, free tier
  2. Local LLM via Open WebUI (OpenAI-compatible API) — private, no rate limits

Configure the local LLM by adding these to your .env file:
  LOCAL_LLM_BASE_URL=http://localhost:3000/api   # Open WebUI base URL
  LOCAL_LLM_API_KEY=your-openwebui-api-key        # Settings -> Account -> API Keys
  LOCAL_LLM_MODEL=llama3.2:latest                 # Model name as shown in Open WebUI

Two public functions:
  get_vibe_params(user_prompt, api_key, playlist_context=None)
      -> DJDirectives for a fresh request

  get_continue_params(original_prompt, previous_queries, api_key, playlist_context=None)
      -> DJDirectives with fresh angles, avoiding the previous queries
"""

from __future__ import annotations

from preferences import build_preference_context, load_preferences

import json
import os
import random
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Gemini models — tried top to bottom, lite first (higher free quota)
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    # gemma-3-4b-it excluded: does not support JSON mode
]

# ---------------------------------------------------------------------------
# Local LLM config — read from .env
# ---------------------------------------------------------------------------
def _read_env() -> dict:
    """Read .env from the project folder."""
    env_path = Path(__file__).parent / ".env"
    result = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip().strip('"').strip("'")
    return result

_ENV = _read_env()

LOCAL_LLM_BASE_URL = _ENV.get("LOCAL_LLM_BASE_URL", os.environ.get("LOCAL_LLM_BASE_URL", ""))
LOCAL_LLM_API_KEY  = _ENV.get("LOCAL_LLM_API_KEY",  os.environ.get("LOCAL_LLM_API_KEY", ""))
LOCAL_LLM_MODEL    = _ENV.get("LOCAL_LLM_MODEL",    os.environ.get("LOCAL_LLM_MODEL", "llama3.2:latest"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INITIAL_PROMPT = """
You are an expert music curator powering a Spotify AI DJ. Your job is to
generate search queries and seed artists that find incredible, specific, and
often undiscovered tracks on Spotify.

User Request: "{user_prompt}"

{playlist_context_block}

---
CRITICAL RULES FOR GOOD QUERIES & DISCOVERY:

1. DEEP DISCOVERY & OBSCURE ARTISTS: The user wants to find artists they would
   NEVER normally have discovered. Do not just recommend mainstream or obvious
   choices. Dig into underground scenes, micro-genres, regional movements,
   independent labels, and hyper-specific niches.
   
2. BEYOND LLM MEMORY (SPOTIFY GRAPH): You may not know every obscure artist.
   To solve this, provide 3-5 real, somewhat niche "seed artists" in the 
   `seed_artists` field. The system will feed these to Spotify's "Related Artists" 
   API, which will surface brand new artists outside your training data.

3. BE SPECIFIC. Vague queries like "dark metal aggressive" return poor results.
   Real artist names, song titles, and known genre terms find real tracks.

4. NON-ENGLISH & NICHE GENRES — this is critical:
   - Search in the ORIGINAL LANGUAGE when relevant. Spotify indexes Cyrillic,
     Chinese, Arabic, Japanese etc. natively. "Любэ" finds more than "Lyube".
   - Also include transliterated versions as separate queries for safety.
   - For regional/folk/cultural music, name the specific tradition, region,
     or movement. "Soviet military choir", "красная армия хор", "Russian romance".
   - KNOW YOUR SCENES: if someone references a specific song or artist,
     identify the genre/scene they belong to and find related artists in it.

5. PLAYLIST INFLUENCE (If provided): 
   - The user's existing taste is provided above. 
   - DO NOT just recommend more of the same. Use it as a compass to point 
     towards adjacent, unexplored, or contrasting scenes. 
   - Example: If they listen to mainstream pop, suggest underground synth-pop 
     from specific regions. If they like metal, suggest obscure folk-metal from 
     specific countries. Lateral moves, not straight lines.

6. MIX STRATEGIES across your queries:
   - Direct artist names (most reliable): "Ансамбль Александрова"
   - Original-script searches:           "Любэ", "Красная армия"
   - Transliterated versions:            "Lyube", "Krasnaya Armiya"
   - Genre/scene terms:                  "Soviet military songs", "военные песни"
   - Era or movement:                    "советские песни", "WWII Soviet"

7. QUANTITY: Generate 6-10 queries. For niche genres, use more queries
   since individual searches may return fewer results.

8. REQUEST ALWAYS WINS. If the user asks for jazz, play jazz — even if their
   history is all metal. The listener history/playlist is only a tiebreaker 
   or a launching pad for vague requests. Never let it override an explicit 
   genre, artist, or mood.

9. SOUNDTRACKS, SCORES & GAME OSTs — this is critical:
   NEVER search for a film/game title as a standalone query. "Destiny" returns
   songs NAMED Destiny, not the game. "Interstellar" returns a country singer.
   USE SPOTIFY FIELD OPERATORS for precision. 
     artist:"Martin O'Donnell"
     album:"Destiny Original Soundtrack"
   Always prefer field operator queries over plain keyword queries for OSTs.
   Set search_mode to "album" for OST requests.

QUEUE SIZE:
  - Casual listen:      20-30 tracks
  - Background/work:    40-60 tracks
  - Gym/long session:   70-100 tracks
  Default 40 if unclear.

{preference_context}
---
EXAMPLES:

Request: "industrial metal like rammstein"
reasoning: "User requested industrial metal. I will bridge this by suggesting industrial acts with strong traditional metal roots, plus deep cuts from the Neue Deutsche Härte scene and related underground acts. Seed artists will help Spotify find adjacent modern industrial acts."
queries: [
  "Rammstein",
  "Oomph! metal",
  "neue deutsche haerte",
  "Eisbrecher",
  "Meganoidi",
  "Deathstars",
  "industrial metal underground"
]
seed_artists: ["Oomph!", "Deathstars", "Meganoidi"]
queue_size: 50
search_mode: "track"

Request: "japanese city pop 80s"
reasoning: "City pop is a defined Japanese genre. I will use both Japanese script and romanized names, focusing on both well-known and deeper-cut artists to ensure discovery."
queries: [
  "山下達郎",
  "竹内まりや",
  "Tatsuro Yamashita",
  "Mariya Takeuchi",
  "Anri city pop",
  "Miki Matsubara",
  "シティポップ",
  "Toshiki Kadomatsu",
  "80s Japanese pop"
]
seed_artists: ["Miki Matsubara", "Anri", "Toshiki Kadomatsu"]
queue_size: 45
search_mode: "track"

---
Output JSON only. No markdown, no explanation outside the JSON.
The JSON must have exactly these fields:
  "reasoning":   string
  "queries":     array of strings
  "queue_size":  integer
  "search_mode": string  ("track" or "album")
  "seed_artists": array of strings (3-5 real, somewhat niche artists for Spotify's related-artist API)
"""

CONTINUE_PROMPT = """
You are an expert music curator. The user is extending their current listening
session and wants MORE music in the same vibe — but with fresh tracks and NEW 
artists they haven't heard yet in this session, ideally ones they would never 
normally have discovered.

Original request: "{user_prompt}"

Queries already used (DO NOT repeat these — find new angles):
{previous_queries}

Currently playing/queued tracks:
{queue_tracks}

{playlist_context_block}

---
YOUR JOB:
Generate 6-10 NEW search queries and 3-5 seed artists that explore DIFFERENT 
angles from those already used, while staying true to the original vibe. 

Good strategies for finding new angles & deep discovery:
  - Artists similar to ones already searched but not yet used (use seed_artists)
  - A different era of the same genre (e.g. 80s vs 90s vs modern)
  - A related subgenre not yet explored (micro-genres, underground scenes)
  - Less mainstream / deeper cuts, B-sides, regional labels
  - Cross-genre fusion (e.g. "jazz metal", "electronic punk")
  - For non-English genres: try different script variants not yet used
  - Regional variations within the same tradition
  - Compilations or anthology searches ("best of X", "X collection")

PLAYLIST INFLUENCE (If provided):
  - Use the user's existing taste to make a lateral move. If the current session 
    is drifting, gently pull it back toward an unexplored corner of the user's 
    broader taste.

SAME RULES APPLY:
  - Prefer real artist names over abstract descriptors
  - Be specific — vague queries return poor results
  - Provide 3-5 `seed_artists` so the system can query Spotify's "Related Artists" 
    API to find brand new artists outside your training data.

Queue size: 40 tracks.

Output JSON only. No markdown, no explanation outside the JSON.
The JSON must have exactly these fields:
  "reasoning": string
  "queries": array of strings
  "queue_size": integer
  "seed_artists": array of strings
"""

STOPWORDS = {
    "play", "some", "me", "i", "want", "can", "you",
    "please", "listen", "to", "for", "a", "put", "on", "the",
    "something", "anything", "music", "songs", "tracks", "more",
}


class DJDirectives(BaseModel):
    """Structured output returned by the AI model."""
    reasoning:   str
    queries:     list[str] = Field(default_factory=list)
    queue_size:  int = 40
    search_mode: str = "track"   # "track" (default) or "album" (for OSTs/scores)
    seed_artists: list[str] = Field(
        default_factory=list, 
        description="3-5 real, somewhat niche artists to use as seeds for Spotify's 'related artists' API. This helps discover new artists beyond the LLM's memory."
    )


def _keyword_fallback(user_prompt: str) -> DJDirectives:
    words   = user_prompt.lower().split()
    cleaned = [w for w in words if w not in STOPWORDS]
    query   = " ".join(cleaned) or user_prompt
    return DJDirectives(
        reasoning="No AI available. Using keyword fallback.",
        queries=[query],
        queue_size=40,
    )


def _parse_local_response(text: str) -> DJDirectives | None:
    """
    Parse a JSON response from a local LLM.
    Local models don't always respect JSON-only output, so we extract
    the first JSON object found in the response text.
    """
    try:
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Find the first { ... } block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            print("[brain:local] No JSON object found in response")
            return None

        data = json.loads(text[start:end])
        d    = DJDirectives(
            reasoning=data.get("reasoning", "Local LLM response"),
            queries=data.get("queries", []),
            queue_size=int(data.get("queue_size", 40)),
            search_mode=data.get("search_mode", "track"),
            seed_artists=data.get("seed_artists", []),
        )
        if not d.queries:
            return None
        d.queue_size = max(1, min(100, d.queue_size))
        return d

    except Exception as e:
        print(f"[brain:local] Parse error: {e}")
        return None


def _call_local_llm(prompt: str) -> DJDirectives | None:
    """
    Call the local LLM via Open WebUI's OpenAI-compatible API.
    Returns None if not configured or if the call fails.
    """
    if not LOCAL_LLM_BASE_URL:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("[brain:local] openai package not installed — run: pip install openai")
        return None

    try:
        client = OpenAI(
            base_url=f"{LOCAL_LLM_BASE_URL.rstrip('/')}/v1",
            api_key=LOCAL_LLM_API_KEY or "local",  # Open WebUI needs a key; use "local" if none set
        )

        response = client.chat.completions.create(
            model=LOCAL_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )

        text = response.choices[0].message.content
        print(f"[brain:local] {LOCAL_LLM_MODEL} responded ({len(text)} chars)")
        return _parse_local_response(text)

    except Exception as e:
        print(f"[brain:local] {LOCAL_LLM_MODEL} error: {e}")
        return None


def _call_gemini(prompt: str, api_key: str) -> DJDirectives | None:
    """Try each Gemini model in order. Returns None if all fail."""
    client = genai.Client(api_key=api_key)

    for model_name in CANDIDATE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DJDirectives,
                ),
            )
            if response.parsed:
                d = response.parsed
                d.queue_size = max(1, min(100, d.queue_size))
                if d.queries:
                    print(f"[brain] {model_name} responded")
                    return d
        except Exception as e:
            err = str(e)
            if "429" in err or "404" in err:
                continue
            print(f"[brain] {model_name}: {err}")
            continue

    return None


def _call_ai(prompt: str, api_key: str, local_only: bool = False) -> DJDirectives | None:
    """
    Try AI models in priority order:
      1. Gemini (if api_key is set AND local_only is False)
      2. Local LLM via Open WebUI / Ollama (if configured)
    Returns None if everything fails.
    """
    # Try Gemini first (unless user wants local only)
    if api_key and not local_only:
        result = _call_gemini(prompt, api_key)
        if result:
            return result
        print("[brain] All Gemini models failed, trying local LLM...")
    elif local_only:
        print("[brain] Local-only mode — skipping Gemini")

    # Try local LLM
    if LOCAL_LLM_BASE_URL:
        result = _call_local_llm(prompt)
        if result:
            print(f"[brain] Local LLM ({LOCAL_LLM_MODEL}) succeeded")
            return result
        print(f"[brain] Local LLM failed")
    else:
        print("[brain] No LOCAL_LLM_BASE_URL set — configure it in Settings")

    return None


PLAYLIST_PROMPT = """
You are an expert music curator powering a Spotify AI DJ.

The user has a Spotify playlist they love. Your job is to analyse the artists
and tracks in that playlist and generate search queries that will find MORE
music with a similar vibe — specifically focusing on NEW artists the user would
never normally have discovered, including those outside your typical training data.

User's extra instructions: "{user_prompt}"

Playlist contents (sample of up to 30 tracks):
{track_list}

---
YOUR TASK:
1. Identify the genre(s), mood, scene, era, and cultural context of this playlist
2. Identify the KEY ARTISTS that define the playlist's sound
3. Generate 8-12 search queries that will find similar music NOT already in the playlist
4. Provide 3-5 "seed artists" in the `seed_artists` field. These should be real, 
   somewhat niche artists in this vibe. The system will use Spotify's "Related 
   Artists" API on these seeds to discover brand new artists outside your memory.

STRATEGIES FOR DEEP DISCOVERY:
- Search for artists who are similar to but distinct from the playlist artists
- Search for the genre/scene/era more broadly to catch artists you might not know
- Include non-English queries if the playlist has non-English content (original script + transliteration)
- Think about: same genre different era, same era different genre, same cultural
  scene, artists who often appear on the same compilations, collaborative artists
- Micro-genres, underground labels, regional variations

CRITICAL: Be specific. Real artist names and known genre terms outperform
abstract descriptors on Spotify search.

Output JSON only. No markdown, no explanation outside the JSON.
Fields: 
  "reasoning" (string)
  "queries" (array of strings)
  "queue_size" (integer, 40-80)
  "seed_artists" (array of 3-5 strings)
"""


def get_playlist_vibe_params(
    playlist_tracks: list[dict],
    user_prompt: str,
    api_key: str,
    local_only: bool = False,
) -> "DJDirectives":
    """
    Analyse a playlist's contents and generate queries for similar music.
    playlist_tracks: raw track dicts from Spotify (must have 'name' and 'artists').
    """
    sample = list(playlist_tracks)
    random.shuffle(sample)
    sample = sample[:30]

    lines = []
    for t in sample:
        artist = t["artists"][0]["name"] if t.get("artists") else "Unknown"
        lines.append(f"  - {t['name']} — {artist}")
    track_list = "\n".join(lines)

    prompt = PLAYLIST_PROMPT.format(
        user_prompt=user_prompt or "Find music that fits alongside these tracks, focusing on deep discovery and new artists.",
        track_list=track_list,
    )

    result = _call_ai(prompt, api_key, local_only=local_only)
    if result:
        return result

    # Fallback: extract unique artists from the playlist and search for them
    artists = list({
        t["artists"][0]["name"]
        for t in playlist_tracks
        if t.get("artists")
    })
    random.shuffle(artists)
    fallback_queries = artists[:8]
    return DJDirectives(
        reasoning="AI unavailable. Falling back to searching for playlist artists directly.",
        queries=fallback_queries,
        queue_size=50,
        seed_artists=artists[:5],
    )


def get_vibe_params(
    user_prompt: str, 
    api_key: str, 
    local_only: bool = False,
    playlist_context: list[str] | str | None = None
) -> DJDirectives:
    """
    Convert a fresh music request into search queries.
    playlist_context: Optional list or string of user's playlist artists/genres 
                      to influence lateral discovery.
    """
    prefs = load_preferences()
    pref_ctx = build_preference_context(prefs)
    
    if playlist_context:
        if isinstance(playlist_context, list):
            ctx_str = ", ".join(playlist_context[:15])  # Top 15 artists/genres
        else:
            ctx_str = playlist_context
        playlist_block = f"User's Existing Taste (Use as a launching pad for lateral discovery, NOT to repeat): {ctx_str}"
    else:
        playlist_block = ""

    prompt = INITIAL_PROMPT.format(
        user_prompt=user_prompt,
        preference_context=pref_ctx,
        playlist_context_block=playlist_block,
    )
    result = _call_ai(prompt, api_key, local_only=local_only)
    if result:
        return result
    return DJDirectives(
        reasoning="All AI models unavailable. Using keyword fallback.",
        queries=_keyword_fallback(user_prompt).queries,
        queue_size=40,
    )


def get_continue_params(
    original_prompt: str,
    previous_queries: list[str],
    api_key: str,
    local_only: bool = False,
    queue_tracks: list[dict] | None = None,
    playlist_context: list[str] | str | None = None,
) -> DJDirectives:
    """
    Generate fresh queries to extend an existing session.
    queue_tracks: list of {name, artist} dicts from the current Spotify queue.
    playlist_context: list or string of user's playlist artists/genres to influence lateral discovery.
    """
    formatted_queries = "\n".join(f"  - {q}" for q in previous_queries)

    if queue_tracks:
        formatted_queue = "\n".join(
            f"  - {t['name']} by {t['artist']}" for t in queue_tracks[:20]
        )
    else:
        formatted_queue = "  (no queue data available — go by original request)"

    if playlist_context:
        if isinstance(playlist_context, list):
            ctx_str = ", ".join(playlist_context[:15])
        else:
            ctx_str = playlist_context
        playlist_block = f"User's Existing Taste (Use for lateral discovery): {ctx_str}"
    else:
        playlist_block = ""

    prompt = CONTINUE_PROMPT.format(
        user_prompt=original_prompt,
        previous_queries=formatted_queries,
        queue_tracks=formatted_queue,
        playlist_context_block=playlist_block,
    )
    result = _call_ai(prompt, api_key, local_only=local_only)
    if result:
        result.queue_size = 40
        return result
    return DJDirectives(
        reasoning="AI unavailable for continue. Repeating original queries.",
        queries=previous_queries,
        queue_size=40,
    )
