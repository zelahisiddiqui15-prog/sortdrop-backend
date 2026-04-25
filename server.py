import os
import json
import anthropic
import stripe
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import voyageai
import re as _re

VOYAGE_CLIENT = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
EMBED_MODEL = "voyage-4-lite"
EMBED_DIMENSION = 512
from models import (init_db, create_user, get_user, get_user_by_email,
                    get_user_by_username, username_exists, get_db,
                    increment_sorts, activate_subscription,
                    deactivate_subscription, set_stripe_customer, hash_password)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Cratify API"})


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    username = data.get("username")

    if not email:
        return jsonify({"error": "email required"}), 400

    existing = get_user_by_email(email)
    if existing:
        return jsonify({"error": "email already registered"}), 409

    if username and username_exists(username):
        return jsonify({"error": "username already taken"}), 409

    user_id = create_user(email=email, username=username, password=password)
    return jsonify({
        "user_id": user_id,
        "username": username,
        "sorts_remaining": 25,
        "subscription_active": False
    })


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    identifier = data.get("email") or data.get("identifier")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"error": "email/username and password required"}), 400

    user = get_user_by_email(identifier)
    if not user:
        user = get_user_by_username(identifier)

    if not user:
        return jsonify({"error": "invalid credentials"}), 401

    if user.get("password_hash") != hash_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])
    return jsonify({
        "user_id": user["id"],
        "username": user.get("username"),
        "email": user["email"],
        "sorts_remaining": sorts_remaining if not user["subscription_active"] else None,
        "subscription_active": bool(user["subscription_active"])
    })


@app.route("/auth/check-username", methods=["GET"])
def check_username():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "username required"}), 400
    taken = username_exists(username)
    return jsonify({"available": not taken, "username": username})


@app.route("/subscription/status", methods=["GET"])
def subscription_status():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    sorts_remaining = None
    if not user["subscription_active"]:
        sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])

    return jsonify({
        "subscription_active": bool(user["subscription_active"]),
        "sorts_used": user["sorts_used"],
        "sorts_remaining": sorts_remaining,
        "trial_limit": user["trial_limit"],
        "username": user.get("username"),
        "email": user.get("email")
    })


@app.route("/classify", methods=["POST"])
def classify():
    data = request.json or {}
    filename = data.get("filename")
    user_id = data.get("user_id")

    if not filename or not user_id:
        return jsonify({"error": "filename and user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    if not user["subscription_active"]:
        if user["sorts_used"] >= user["trial_limit"]:
            return jsonify({"error": "trial_exhausted"}), 402

    prompt = f"""You are a music file classifier for a producer tool called Cratify.

Analyze this filename and return a JSON object with these fields:
- category: Bass, Lead, Pad, Pluck, FX, Drum, Vocal, Chord, Arp, Guitar, Piano, Strings, Brass, Synth, Texture, Ambient, or Other.
- drum_type: ONLY for Drum: Kick, Snare, Hi-Hat, Clap, Perc, Cymbal, Tom, Full Loop. Null for non-drums.
- subcategory: more specific description
- key: musical key if detectable (e.g. "Am", "C#") or null. Always null for drums.
- bpm: BPM if detectable as number or null
- file_type: "stem", "preset", "midi", "sample", or "loop"
- confidence: 0 to 1

CRITICAL CATEGORY RULES:
Never use Loop as a standalone category. Instead classify by the instrument type:
- Drum Loop, Beat, Break → category: Drum (set drum_type: "Full Loop")
- Bass Loop → category: Bass
- Synth Loop, Synth Riff → category: Synth
- Piano Loop → category: Piano
- Guitar Loop → category: Guitar
- Chord Loop, Chord Stab → category: Chord
- Melody Loop, Lead Loop → category: Lead
- Vocal Loop, Vox Loop → category: Vocal
- Arp Loop → category: Arp
- Pad Loop, Atmosphere Loop → category: Pad
- FX Loop, Riser, Sweep → category: FX
- Texture Loop, Ambient Loop → category: Ambient
- If a loop's instrument is truly unknown: category: Other

Set file_type to "loop" whenever the filename contains "loop", "lp", "riff", "break", or "beat".

Filename: {filename}

Return ONLY valid JSON. No markdown, no explanation."""

    try:
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as api_err:
        print(f"[classify] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": str(api_err)}), 500

    try:
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception:
        result = {
            "category": "Other", "drum_type": None, "subcategory": "Unknown",
            "key": None, "bpm": None, "file_type": "stem", "confidence": 0.5
        }

    # Post-process: drums should never have keys
    if result.get("category") == "Drum":
        result["key"] = None
    # Force preset category for preset file extensions
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if ext in ('fxp', 'vital', 'nmsv', 'xpf', 'aupreset', 'patch'):
        result['category'] = 'Preset'
        result['key'] = None
        result['bpm'] = None
        result['file_type'] = 'preset'
    increment_sorts(user_id)
    return jsonify(result)


@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["client_reference_id"]
        customer_id = session["customer"]
        subscription_id = session["subscription"]
        if user_id:
            set_stripe_customer(user_id, customer_id)
            activate_subscription(customer_id, subscription_id)
    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        deactivate_subscription(sub["customer"])

    return jsonify({"status": "ok"})


@app.route("/stripe/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.json or {}
    user_id = data.get("user_id")
    try:
        session = stripe.checkout.Session.create(
            ui_mode="embedded_page",
            line_items=[{"price": os.getenv("STRIPE_PRICE_ID"), "quantity": 1}],
            mode="subscription",
            return_url=f"https://www.cratify.app/dashboard?session_id={{CHECKOUT_SESSION_ID}}",
            client_reference_id=user_id,
        )
        return jsonify({"clientSecret": session.client_secret})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_INTENT_SYSTEM = """You parse music producer chat messages into structured bulk file actions.

Return ONLY a valid JSON object — no explanation, no markdown fences.

Supported actions: "export" (copy files to a destination folder) or "move" (move files).
Return action: null if the message is just a search or question, not a bulk operation.

Output schema:
{
  "action": "export" | "move" | null,
  "filter": {
    "key":      string | null,   // musical key, e.g. "C# minor", "Am", "G major"
    "category": string | null,   // e.g. "loop", "bass", "drum", "vocal", "pad"
    "bpm_min":  number | null,
    "bpm_max":  number | null,
    "file_type": string | null   // extension without dot: "wav", "mp3", "midi"
  },
  "destination": string | null   // absolute path the user mentioned, or null
}

Decision rules:
- "export / send / copy … to …"  → action: "export"
- "move … to …"                  → action: "move"
- "show / find / search / what"  → action: null
- Vague questions with no clear destination → action: null
- If destination folder is not explicitly stated → destination: null

Examples:
  "export all my C# minor loops to /Users/zee/Desktop"
  → {"action":"export","filter":{"key":"C# minor","category":"loop","bpm_min":null,"bpm_max":null,"file_type":null},"destination":"/Users/zee/Desktop"}

  "move all bass wav files to /Users/zee/Music/Project"
  → {"action":"move","filter":{"key":null,"category":"bass","bpm_min":null,"bpm_max":null,"file_type":"wav"},"destination":"/Users/zee/Music/Project"}

  "send everything between 120 and 130 bpm to my desktop"
  → {"action":"export","filter":{"key":null,"category":null,"bpm_min":120,"bpm_max":130,"file_type":null},"destination":"/Users/zee/Desktop"}

  "show me all loops in Am"
  → {"action":null,"filter":{},"destination":null}

  "find dark pads in C minor"
  → {"action":null,"filter":{},"destination":null}
"""


@app.route("/intent", methods=["POST"])
def intent():
    """Parse a chat message for bulk file action intent using Claude Haiku."""
    data = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"action": None})
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": message}]
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if model wraps anyway
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rsplit("```", 1)[0]
        result = json.loads(raw.strip())
        # Normalise: always return the top-level action key
        if "action" not in result:
            result["action"] = None
        return jsonify(result)
    except Exception as e:
        print(f"[/intent] error: {e}", flush=True)
        return jsonify({"action": None})


_PAIR_MAP = {
    "kick":      ["snare", "clap", "hi-hat", "hihat"],
    "bass":      ["pad", "lead", "pluck"],
    "lead":      ["pad", "arp"],
    "loop":      ["drum loop", "bass"],
}

@app.route("/pair", methods=["POST"])
def pair():
    """Given a filepath + category, return top 5 complementary files from target categories."""
    data = request.json or {}
    filepath = data.get("filepath", "").strip()
    category = (data.get("category") or "").strip().lower()

    if not filepath or not category:
        return jsonify({"error": "filepath and category required"}), 400

    # Determine target categories from pairing map
    target_cats = None
    for key, targets in _PAIR_MAP.items():
        if key in category:
            target_cats = targets
            break
    if not target_cats:
        return jsonify({"error": f"no pairing defined for category '{category}'"}), 400

    try:
        import sqlite3 as _sqlite3
        import numpy as _np
        import base64 as _b64
        from pathlib import Path as _Path
        from scipy.spatial.distance import cosine as _cosine

        db_path = str(_Path.home() / ".cratify" / "index.db")

        # Load embedding for the query file
        conn = _sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT embedding FROM files WHERE filepath = ?", (filepath,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0]:
            return jsonify({"error": "no embedding found for this file — run indexer first"}), 404

        query_emb = _np.frombuffer(row[0], dtype=_np.float32)

        # Build SQL LIKE clause for target categories (case-insensitive)
        placeholders = " OR ".join(["LOWER(category) LIKE ?" for _ in target_cats])

        conn2 = _sqlite3.connect(db_path)
        c2 = conn2.cursor()
        c2.execute(
            f"SELECT filepath, filename, category, key, bpm, embedding FROM files "
            f"WHERE embedding IS NOT NULL AND filepath != ? AND ({placeholders})",
            [filepath] + [f"%{t}%" for t in target_cats],
        )
        candidates = c2.fetchall()
        conn2.close()

        results = []
        for fp, fn, cat, key, bpm, emb_blob in candidates:
            try:
                emb = _np.frombuffer(emb_blob, dtype=_np.float32)
                sim = float(1.0 - _cosine(query_emb, emb))
                results.append({
                    "filepath": fp,
                    "filename": fn,
                    "category": cat or "",
                    "key": key or "",
                    "bpm": bpm,
                    "similarity": round(sim, 4),
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return jsonify({"pairs": results[:5], "target_categories": target_cats})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/embed", methods=["POST"])
def embed():
    """Generate text embeddings via Voyage AI for semantic search."""
    data = request.get_json(force=True) or {}
    texts = data.get("texts", [])
    input_type = data.get("input_type", "document")

    if not texts:
        return jsonify({"embeddings": []})

    BATCH = 128
    all_embeddings = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        try:
            result = VOYAGE_CLIENT.embed(
                chunk,
                model=EMBED_MODEL,
                input_type=input_type,
                output_dimension=EMBED_DIMENSION,
            )
            all_embeddings.extend(result.embeddings)
        except Exception as e:
            print(f"[embed] chunk {i} failed: {e}", flush=True)
            return jsonify({"error": str(e)}), 500

    return jsonify({
        "embeddings": all_embeddings,
        "model": EMBED_MODEL,
        "dimension": EMBED_DIMENSION,
    })


@app.route("/summarize_project", methods=["POST"])
def summarize_project():
    """Generate a 1-liner project summary for the sidebar.

    Notes are the source of truth (the project brief).
    Chat history layers on evolution — but dominant patterns win
    over one-off tangents.

    Request body:
      {
        "notes": "G minor sad/melancholic with two drops" (str),
        "messages": [
          {"role": "user", "content": "find me an arp"},
          {"role": "assistant", "content": "..."},
          ...
        ]  (list, last ~20 messages, can be empty)
      }

    Response:
      { "summary": "G min · 128 BPM · dark/melancholic" }

    Cost: ~$0.0015 per call (Haiku 4.5).
    """
    data = request.get_json(force=True) or {}
    notes = (data.get("notes") or "").strip()
    messages = data.get("messages", [])

    # If we have neither notes nor messages, return empty — nothing
    # to summarize. Client falls back to first line of notes (which
    # is also empty in this case, so sidebar shows nothing).
    if not notes and not messages:
        return jsonify({"summary": ""})

    # Build the chat transcript section. Cap at last 20 messages
    # to keep token count predictable. Skip empty messages.
    transcript_lines = []
    for m in messages[-20:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        # Trim each message to ~200 chars to stay focused on intent
        if len(content) > 200:
            content = content[:200] + "..."
        prefix = "User" if role == "user" else "AI"
        transcript_lines.append(f"{prefix}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no chat history yet)"

    notes_block = notes if notes else "(no notes provided)"

    system_prompt = """You are a music producer's assistant generating a glanceable 1-line summary of a song project for a sidebar UI.

The producer needs to scan a list of 5-15 active song projects and immediately remember what each one is about. Your output is the entire 1-liner shown under the project name.

YOU WILL RECEIVE:
1. PROJECT NOTES — the producer's explicit brief (highest priority, treat as source of truth)
2. RECENT CHAT HISTORY — messages between the producer and the AI inside this project (use to surface dominant patterns, NOT to chase recent tangents)

CRITICAL RULES:
- NOTES ARE THE ANCHOR. If notes say "G minor", the summary stays G minor even if recent messages mention other keys.
- DOMINANT PATTERNS WIN. If user discussed key X across 8 messages and key Y in 1 message, summary uses key X. Tangents do not shift the summary.
- BE TERSE. Producer shorthand only. No adjectives like "really" / "kind of" / "very" / "with".
- STRICT FORMAT: <key> · <BPM> · <2-3 vibe tags>
- USE BULLET SEPARATOR: · (middle dot, U+00B7) between sections
- MAX 8 WORDS TOTAL
- Use producer key shorthand: "Gm" not "G minor", "F#" not "F sharp", "Am" not "A minor", etc.
- BPM: integer or short range like "128" or "120-130"
- Vibe tags: 1-3 short tags like "dark", "melancholic", "uplifting", "trap", "dubstep", "afro house"

EXAMPLES:
"Gm · 128 BPM · dark trap"
"F# · 140 BPM · dubstep, melancholic"
"Am · 90/180 BPM · afro house"
"Em · ~125 BPM · progressive, uplifting"

If you cannot determine a section, omit it. E.g. if no key is clear: "128 BPM · dark trap". If only key is clear: "Gm · melancholic".

If neither notes nor chat history give you anything to work with, return an empty string.

Respond with ONLY the 1-liner. No quotes, no explanation, no preamble."""

    user_prompt = f"""PROJECT NOTES:
{notes_block}

RECENT CHAT HISTORY (last 20 messages, oldest first):
{transcript}

Generate the 1-liner."""

    try:
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        print(f"[summarize_project] claude call failed: {e}", flush=True)
        return jsonify({"error": f"claude_failed: {e}"}), 500

    # Extract text from the response. Haiku returns text content blocks.
    summary = ""
    for block in response.content:
        if hasattr(block, "text"):
            summary += block.text
    summary = summary.strip()

    # Defensive cleanup: strip surrounding quotes if Claude added them,
    # collapse whitespace, cap length so a misbehaving response can't
    # blow out the sidebar.
    summary = summary.strip('"').strip("'").strip()
    summary = " ".join(summary.split())  # collapse internal whitespace
    if len(summary) > 80:
        summary = summary[:77] + "..."

    print(f"[summarize_project] summary: {summary!r}", flush=True)
    return jsonify({"summary": summary})


@app.route("/search", methods=["POST"])
def search():
    """Semantic search: client sends top-50 candidates by cosine similarity,
    Claude re-ranks, writes an explanation, and returns structured filters."""
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    candidates = data.get("candidates", [])
    conversation = data.get("conversation", [])

    if not query:
        return jsonify({"error": "empty query"}), 400

    if not candidates:
        return jsonify({
            "picks": [],
            "filters_used": {},
            "reply": "Your library has no samples indexed yet.",
            "broad_count": 0,
        })

    candidates_text = "\n".join(
        f"[{c['id']}] {c['meta_text']}"
        for c in candidates[:50]
    )

    system_prompt = """You are Cratify, an AI music producer assistant. You help producers find the perfect sample from their personal library.

The user will ask for sounds. You have been given the top-50 most semantically similar samples from their library, pre-ranked by vector similarity. Your job is to analyze them, identify the best matches, write a concise producer-friendly explanation, and infer the broader filter criteria for a "see more" button.

Guidelines for your response:
- picks: identify the 4-8 best actual matches. If fewer than 4 truly match, return fewer. If nothing matches well, return empty list.
- reply: write like a text message to a producer friend. Max 3-4 short sentences across 2-3 short paragraphs separated by blank lines. NO run-on sentences. Mention relative major/minor when relevant, pitch-adjust tolerances, layering advice.
- NEVER mention the [ID] numbers in your reply text. Refer to samples by filename or short descriptor ("the F wub one-shot", "that Cm kick").
- filters_used: describes the broader search the user might want ("all vocal chops in Gm around 140 BPM") - be permissive, it's an escape hatch. Any field can be omitted if not inferrable.
- category MUST be one of: Drums, Bass, Synth, Leads, Vocals, FX, Loops, One-Shots, Keys, Percussion, Other

You MUST respond by calling the return_search_results tool. Do not respond with text.

The user's library samples (pre-ranked by similarity, ID in brackets):
""" + candidates_text

    SEARCH_TOOL_SCHEMA = {
        "name": "return_search_results",
        "description": "Return ranked sample picks with an explanation and inferred filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "description": "Best matching samples, 0-8 items, ranked by relevance.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Sample ID from the candidates list."},
                            "score": {"type": "number", "description": "Match quality 0.0-1.0."},
                            "reason": {"type": "string", "description": "Short phrase explaining why this matches."},
                        },
                        "required": ["id", "score", "reason"],
                    },
                },
                "reply": {
                    "type": "string",
                    "description": "Producer-friendly explanation, 2-3 short paragraphs separated by blank lines.",
                },
                "filters_used": {
                    "type": "object",
                    "description": "Broader filter criteria the user might want (for 'see more' button).",
                    "properties": {
                        "category": {"type": "string"},
                        "key": {"type": "string"},
                        "bpm_min": {"type": "integer"},
                        "bpm_max": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["picks", "reply", "filters_used"],
        },
    }

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = conversation + [{"role": "user", "content": query}]

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system_prompt,
            messages=messages,
            tools=[SEARCH_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "return_search_results"},
        )
    except Exception as e:
        print(f"[search] claude call failed: {e}", flush=True)
        return jsonify({"error": f"claude_failed: {e}"}), 500

    # Extract the tool_use block. With tool_choice forcing this tool, Claude
    # MUST return a tool_use block with validated input matching the schema.
    tool_use = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "return_search_results":
            tool_use = block
            break

    if tool_use is None:
        print(f"[search] no tool_use block returned. Content: {response.content}", flush=True)
        return jsonify({
            "picks": [],
            "filters_used": {},
            "reply": "Sorry, I had trouble formatting that response. Try rephrasing your search.",
            "broad_count": 0,
            "error": "no_tool_use",
        })

    parsed = tool_use.input  # guaranteed dict matching schema

    return jsonify({
        "picks": parsed.get("picks", []),
        "filters_used": parsed.get("filters_used", {}),
        "reply": parsed.get("reply", ""),
        "broad_count": 0,
    })


if __name__ == "__main__":
    import sys
    print("Starting Cratify API...", flush=True)
    try:
        init_db()
        print("Database initialized", flush=True)
    except Exception as e:
        print(f"Database error: {e}", flush=True)
        sys.exit(1)
    port = int(os.getenv("PORT", 5000))
    print(f"Running on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
