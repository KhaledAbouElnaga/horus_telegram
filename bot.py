import os
import base64
import asyncio
import hashlib
import tempfile
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

import imageio_ffmpeg
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI, RateLimitError, APIError
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_USER_ID = int(os.getenv("ALLOWED_TELEGRAM_USER_ID", "0"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.6-terra").strip()
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").strip()
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-transcribe").strip()
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "cedar").strip()
AUDIO_MODEL = os.getenv("OPENAI_AUDIO_MODEL", "gpt-audio-1.5").strip()
AUDIO_VOICE = os.getenv("OPENAI_AUDIO_VOICE", "cedar").strip()

DEFAULT_VOICE_MODE = os.getenv("DEFAULT_VOICE_MODE", "eco").strip().lower()
DEBUG_TEXT = os.getenv("DEBUG_TEXT", "false").strip().lower() == "true"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not ALLOWED_USER_ID:
    raise RuntimeError("Missing ALLOWED_TELEGRAM_USER_ID")
if DEFAULT_VOICE_MODE not in {"eco", "natural"}:
    DEFAULT_VOICE_MODE = "eco"

client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_FILE = Path(__file__).with_name("horus_system_prompt.txt")
if not PROMPT_FILE.exists():
    raise RuntimeError("horus_system_prompt.txt not found")

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

TTS_INSTRUCTIONS = """
Speak as a native Egyptian Arabic male speaker from Cairo.
Egyptian Arabic must sound like your first language, not a learned accent.

Use natural everyday Cairene rhythm, stress, pacing, reductions and intonation.
Do not use Modern Standard Arabic pronunciation unless a formal phrase requires it.
Do not sound Gulf, Levantine, foreign, theatrical, like a presenter, narrator,
teacher, or customer-service agent. Do not over-enunciate Arabic consonants.

Use normal Cairene pronunciation naturally. In casual Egyptian words, use the
pronunciation a Cairo native would normally use, including common Cairene
realizations of ق, ج, ث and ذ when appropriate, without exaggeration.

Speak at a relaxed normal conversational pace with natural short pauses.
The listener is Khaled. The speaker is Horus.
Sound like a smart Egyptian male personal assistant talking to someone he knows well.

For Egyptian words such as:
عايز، عايزك، إيه، كده، دلوقتي، مش، مفيش، لسه، عشان، إحنا، هنعمل، برده، أوي
pronounce them naturally as everyday Cairene Egyptian.
Do not perform an Egyptian accent. Simply sound native.
"""

DIRECT_AUDIO_PROMPT = SYSTEM_PROMPT + """
\nDIRECT VOICE MODE:
Listen to Khaled's actual audio carefully. He is a native Egyptian Arabic speaker.
He may speak quickly, shorten words, swallow sounds, mix Arabic and English,
use slang, or change direction mid-sentence.

Understand his speech directly as Egyptian Arabic. Do not rewrite it mentally
into formal Arabic first.

Your audio reply must sound like a native Egyptian man from Cairo:
natural, direct, relaxed and conversational. Avoid foreign-learner rhythm,
formal MSA delivery, presenter voice, customer-service voice and exaggerated slang.
Usually keep the answer concise and do not repeat Khaled's question.
"""

TRANSCRIPTION_PROMPT = """
The speaker is Khaled, a native Egyptian Arabic speaker.
Transcribe what he actually says in colloquial Egyptian Arabic.
Preserve Egyptian wording and Arabic/English code-switching.
Do not rewrite his speech into Modern Standard Arabic.

Recurring terms:
Khaled / خالد
Horus / حورس
MOMENT / مومنت
Jeebly / جيبلي
Absher / أبشر / Absher Group
Gmail / Google Drive / Google Sheets / Google Docs / Google Calendar / Google Tasks

Common Egyptian words can include:
عايز، عايزك، إيه، كده، دلوقتي، برده، لسه، عشان، مفيش، مش، إحنا، هعمل، هنعمل، بص
"""

histories = {}
voice_modes = {}
MAX_HISTORY = 12


def authorized(user_id: int) -> bool:
    return user_id == ALLOWED_USER_ID


def history(user_id: int):
    return histories.setdefault(user_id, [])


def remember(user_id: int, role: str, content: str):
    if not content:
        return
    history(user_id).append({"role": role, "content": content})
    histories[user_id] = history(user_id)[-MAX_HISTORY:]


def get_mode(user_id: int) -> str:
    return voice_modes.get(user_id, DEFAULT_VOICE_MODE)


def set_mode(user_id: int, mode: str):
    voice_modes[user_id] = mode


def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=f,
            language="ar",
            prompt=TRANSCRIPTION_PROMPT,
        )
    return (result.text or "").strip()


def build_context(user_id: int, new_text: str) -> str:
    parts = []
    for item in history(user_id)[-MAX_HISTORY:]:
        who = "Khaled" if item["role"] == "user" else "Horus"
        parts.append(f"{who}: {item['content']}")
    parts.append(f"Khaled: {new_text}")
    parts.append("Horus:")
    return "\n\n".join(parts)


def ask_text(user_id: int, text: str) -> str:
    response = client.responses.create(
        model=TEXT_MODEL,
        instructions=SYSTEM_PROMPT,
        input=build_context(user_id, text),
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=700,
    )
    answer = (response.output_text or "").strip()
    if not answer:
        answer = "معلش، الرد مطلعش مظبوط. قولهالي تاني."
    remember(user_id, "user", text)
    remember(user_id, "assistant", answer)
    return answer


def make_eco_voice(text: str, path: str):
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=TTS_INSTRUCTIONS,
        response_format="opus",
        speed=1.0,
    ) as response:
        response.stream_to_file(path)


def to_wav(src: str, dst: str):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-ac", "1", "-ar", "24000",
         "-c:a", "pcm_s16le", dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def wav_to_ogg(src: str, dst: str):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-c:a", "libopus", "-b:a", "64k", dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def ask_natural_audio(user_id: int, wav_path: str):
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [{"role": "developer", "content": DIRECT_AUDIO_PROMPT}]
    messages += [
        {"role": item["role"], "content": item["content"]}
        for item in history(user_id)[-MAX_HISTORY:]
    ]
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "دي أحدث رسالة صوتية من خالد. اسمعها ورد عليه مباشرة بمصري طبيعي.",
            },
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
        ],
    })

    completion = client.chat.completions.create(
        model=AUDIO_MODEL,
        modalities=["text", "audio"],
        audio={"voice": AUDIO_VOICE, "format": "wav"},
        messages=messages,
    )

    msg = completion.choices[0].message
    audio = getattr(msg, "audio", None)
    if not audio or not getattr(audio, "data", None):
        raise RuntimeError("GPT-Audio returned no audio.")

    audio_bytes = base64.b64decode(audio.data)
    transcript = getattr(audio, "transcript", None) or getattr(msg, "content", None) or ""
    return audio_bytes, str(transcript).strip()


def normalize(text: str) -> str:
    return (text.lower()
            .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ى", "ي"))


def mode_from_text(text: str):
    t = normalize(text)
    natural = [
        "فعل الصوت الطبيعي", "شغل الصوت الطبيعي", "الوضع الطبيعي",
        "شغل الوضع الطبيعي", "عايز الصوت الطبيعي", "خلي الصوت طبيعي",
    ]
    eco = [
        "فعل الوضع الاقتصادي", "شغل الوضع الاقتصادي",
        "ارجع للوضع الاقتصادي", "الصوت الاقتصادي",
        "خليها اقتصادي", "خليه اقتصادي", "ارجع اقتصادي",
    ]
    if any(normalize(x) in t for x in natural):
        return "natural"
    if any(normalize(x) in t for x in eco):
        return "eco"
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not authorized(user.id):
        return
    await update.effective_message.reply_text(
        f"حورس شغال.\nالوضع الحالي: {get_mode(user.id)}\nكلمني Voice أو كتابة."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not authorized(user.id):
        return
    histories.pop(user.id, None)
    await update.effective_message.reply_text("تمام، صفّرت سياق المحادثة.")


async def eco(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not authorized(user.id):
        return
    set_mode(user.id, "eco")
    await update.effective_message.reply_text("تمام. الوضع الاقتصادي شغال.")


async def natural(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not authorized(user.id):
        return
    set_mode(user.id, "natural")
    await update.effective_message.reply_text(
        "تمام. الصوت الطبيعي شغال. خلي بالك إن الوضع ده أغلى في استهلاك الـAPI."
    )


async def show_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not authorized(user.id):
        return
    mode = get_mode(user.id)
    text = (
        "الوضع الحالي: Natural — جودة أعلى وتكلفة أعلى."
        if mode == "natural"
        else "الوضع الحالي: Eco — ده الوضع الاقتصادي الافتراضي."
    )
    await update.effective_message.reply_text(text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message = update.effective_message

    if not user or not authorized(user.id):
        return

    media = message.voice or message.audio
    if not media:
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.RECORD_VOICE,
        )

        tg_file = await context.bot.get_file(media.file_id)

        with tempfile.TemporaryDirectory() as td:
            if message.voice:
                suffix = ".ogg"
            else:
                filename = getattr(message.audio, "file_name", None) or "audio.mp3"
                suffix = Path(filename).suffix or ".mp3"

            source = str(Path(td) / f"khaled{suffix}")
            await tg_file.download_to_drive(source)

            transcript = await asyncio.to_thread(transcribe, source)

            if DEBUG_TEXT:
                print("KHALED:", transcript, flush=True)

            if not transcript:
                await message.reply_text("الصوت موصلنيش واضح. ابعته تاني.")
                return

            requested = mode_from_text(transcript)
            if requested:
                set_mode(user.id, requested)
                await message.reply_text(
                    "تمام، شغلت الصوت الطبيعي."
                    if requested == "natural"
                    else "تمام، رجعت للوضع الاقتصادي."
                )
                return

            if get_mode(user.id) == "eco":
                answer = await asyncio.to_thread(ask_text, user.id, transcript)
                out = str(Path(td) / "horus.ogg")
                await asyncio.to_thread(make_eco_voice, answer, out)
                with open(out, "rb") as f:
                    await message.reply_voice(voice=f)
                return

            input_wav = str(Path(td) / "khaled.wav")
            output_wav = str(Path(td) / "horus.wav")
            output_ogg = str(Path(td) / "horus.ogg")

            await asyncio.to_thread(to_wav, source, input_wav)
            audio_bytes, horus_text = await asyncio.to_thread(
                ask_natural_audio, user.id, input_wav
            )

            Path(output_wav).write_bytes(audio_bytes)
            await asyncio.to_thread(wav_to_ogg, output_wav, output_ogg)

            remember(user.id, "user", transcript)
            remember(user.id, "assistant", horus_text)

            if DEBUG_TEXT:
                print("HORUS:", horus_text, flush=True)

            with open(output_ogg, "rb") as f:
                await message.reply_voice(voice=f)

    except RateLimitError:
        await message.reply_text("في مشكلة في الرصيد أو حدود OpenAI API.")
    except APIError as e:
        await message.reply_text(f"OpenAI رجّع Error:\n{e}")
    except Exception as e:
        print(f"VOICE ERROR: {type(e).__name__}: {e}", flush=True)
        await message.reply_text(
            f"حصل خطأ أثناء معالجة الصوت:\n{type(e).__name__}: {e}"
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message = update.effective_message

    if not user or not authorized(user.id):
        return

    text = (message.text or "").strip()
    if not text:
        return

    requested = mode_from_text(text)
    if requested:
        set_mode(user.id, requested)
        await message.reply_text(
            "تمام، شغلت الصوت الطبيعي."
            if requested == "natural"
            else "تمام، رجعت للوضع الاقتصادي."
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )
        answer = await asyncio.to_thread(ask_text, user.id, text)
        await message.reply_text(answer)
    except RateLimitError:
        await message.reply_text("في مشكلة في الرصيد أو حدود OpenAI API.")
    except Exception as e:
        print(f"TEXT ERROR: {type(e).__name__}: {e}", flush=True)
        await message.reply_text(f"حصل خطأ:\n{type(e).__name__}: {e}")


telegram_app = Application.builder().token(BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("reset", reset))
telegram_app.add_handler(CommandHandler("eco", eco))
telegram_app.add_handler(CommandHandler("natural", natural))
telegram_app.add_handler(CommandHandler("mode", show_mode))
telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# Secret webhook path/header derived locally from the bot token.
token_hash = hashlib.sha256(BOT_TOKEN.encode("utf-8")).hexdigest()
WEBHOOK_SECRET = token_hash[:48]
WEBHOOK_PATH = f"/telegram/{token_hash[48:80]}"

background_tasks = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_app.initialize()
    await telegram_app.start()

    external_url = RENDER_EXTERNAL_URL or os.getenv("WEBHOOK_BASE_URL", "").strip()
    if not external_url:
        raise RuntimeError("RENDER_EXTERNAL_URL is missing.")

    webhook_url = f"{external_url.rstrip('/')}{WEBHOOK_PATH}"

    await telegram_app.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=False,
    )

    print("Horus Render Webhook Bot is running.", flush=True)
    print(f"Webhook host: {external_url}", flush=True)
    print(f"Default voice mode: {DEFAULT_VOICE_MODE}", flush=True)

    try:
        yield
    finally:
        # Important: do not delete the webhook here.
        # Render Free can sleep/restart; Telegram must keep pointing to this URL.
        await telegram_app.stop()
        await telegram_app.shutdown()


app = FastAPI(title="Horus Telegram Bot", lifespan=lifespan)


@app.get("/")
async def home():
    return {"ok": True, "service": "Horus", "telegram": "webhook"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)

    # Reply to Telegram immediately. OpenAI/audio work continues in the background.
    task = asyncio.create_task(telegram_app.process_update(update))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    return {"ok": True}
