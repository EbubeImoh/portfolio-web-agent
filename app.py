"""Web chat application that wraps the original workbook logic into Gradio."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from pypdf import PdfReader

load_dotenv(override=True)

LOG_LEVEL_NAME = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL_NAME, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("career_chat")

DATA_DIR = Path(__file__).parent / "data"
SUMMARY_PATH = DATA_DIR / "summary.txt"
PROFILE_PATH = DATA_DIR / "Profile.pdf"
TOOL_LOG_PATH = DATA_DIR / "tool_events.csv"
TOOL_LOG_FIELDS = ["timestamp", "event_type", "payload"]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

NAME = "Ebube Imoh"
MAX_USER_MESSAGE_CHARS = int(os.getenv("MAX_USER_MESSAGE_CHARS", "1000"))
FORBIDDEN_TERMS = tuple(
    term.strip().lower()
    for term in os.getenv("MESSAGE_FORBIDDEN_TERMS", "bomb,explode,weapon").split(",")
    if term.strip()
)


class FileCache:
    """Cache text content from a file and refresh when the file changes."""

    def __init__(self, path: Path, loader: Callable[[Path], str]) -> None:
        self.path = path
        self.loader = loader
        self._cached_value = ""
        self._mtime: float | None = None

    def get(self) -> str:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            logger.error("Required file %s not found.", self.path)
            self._cached_value = ""
            self._mtime = None
            return ""

        if self._mtime != mtime:
            logger.info("Refreshing cached content from %s", self.path.name)
            try:
                self._cached_value = self.loader(self.path)
                self._mtime = mtime
            except Exception:
                logger.exception("Failed to load %s", self.path)
                self._cached_value = ""
                self._mtime = None
        return self._cached_value


def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_summary_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


SUMMARY_CACHE = FileCache(SUMMARY_PATH, load_summary_text)
PROFILE_CACHE = FileCache(PROFILE_PATH, load_pdf_text)


def append_tool_event(event_type: str, payload: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": json.dumps(payload),
    }
    file_exists = TOOL_LOG_PATH.exists()
    try:
        with TOOL_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TOOL_LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        logger.exception("Failed to write tool event to %s", TOOL_LOG_PATH)


def push(message: str) -> None:
    """Send a pushover notification if creds are available."""
    if not (PUSHOVER_USER and PUSHOVER_TOKEN):
        logger.debug("Skipping Pushover notification because credentials are missing.")
        return

    payload = {"user": PUSHOVER_USER, "token": PUSHOVER_TOKEN, "message": message}
    try:
        requests.post(PUSHOVER_URL, data=payload, timeout=10)
    except requests.RequestException:
        logger.exception("Pushover request failed.")


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> Dict[str, str]:
    record = {"email": email, "name": name, "notes": notes}
    append_tool_event("record_user_details", record)
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> Dict[str, str]:
    record = {"question": question}
    append_tool_event("record_unknown_question", record)
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}


record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


def handle_tool_calls(tool_calls: Sequence[ChatCompletionMessageToolCall]) -> List[Dict[str, Any]]:
    """Execute tool calls requested by the LLM and format results."""
    results: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        if not callable(tool):
            logger.error("Unknown tool requested: %s", tool_name)
            continue
        logger.info("Executing tool call: %s", tool_name)
        tool_result = tool(**arguments)
        results.append({"role": "tool", "content": json.dumps(tool_result), "tool_call_id": tool_call.id})
    return results


def compose_system_prompt() -> str:
    summary = SUMMARY_CACHE.get() or "[Summary unavailable]"
    linkedin = PROFILE_CACHE.get() or "[LinkedIn profile unavailable]"

    prompt = (
        f"You are acting as {NAME}. You are answering questions on {NAME}'s website, particularly questions related to "
        f"{NAME}'s career, background, skills and experience. Your responsibility is to represent {NAME} for interactions "
        f"on the website as faithfully as possible. You are given a summary of {NAME}'s background and LinkedIn profile "
        f"which you can use to answer questions. Be professional and engaging, as if talking to a potential client or "
        f"future employer who came across the website. If you don't know the answer to any question, use your "
        f"record_unknown_question tool to record the question that you couldn't answer, even if it's about something "
        f"trivial or unrelated to career. If the user is engaging in discussion, try to steer them towards getting in "
        f"touch via email; ask for their email and record it using your record_user_details tool."
    )
    prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
    prompt += f"With this context, please chat with the user, always staying in character as {NAME}."
    return prompt


def ensure_client() -> OpenAI:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set in the environment.")
    return OpenAI(api_key=GOOGLE_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def history_to_messages(history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages


def run_chat(messages: List[Dict[str, Any]]) -> ChatCompletionMessage:
    """Call Gemini with tool support until a textual response is produced."""
    client = ensure_client()
    done = False
    response_message: ChatCompletionMessage | None = None

    while not done:
        response = client.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls" and response_message and response_message.tool_calls:
            logger.debug("LLM requested tool calls.")
            tool_results = handle_tool_calls(response_message.tool_calls)
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [tc.model_dump() for tc in response_message.tool_calls],
                }
            )
            messages.extend(tool_results)
        else:
            done = True

    assert response_message is not None
    return response_message


def enforce_message_policies(message: str) -> Tuple[bool, str | None]:
    if len(message) > MAX_USER_MESSAGE_CHARS:
        return False, f"Please keep messages under {MAX_USER_MESSAGE_CHARS} characters."
    lowered = message.lower()
    for term in FORBIDDEN_TERMS:
        if term and term in lowered:
            return False, "Your message contains content that cannot be processed."
    return True, None


def chat(message: str, history: List[Tuple[str, str]]) -> str:
    allowed, violation = enforce_message_policies(message)
    if not allowed:
        logger.warning("Blocked user message: %s", violation)
        return violation or ""

    messages: List[Dict[str, Any]] = [{"role": "system", "content": compose_system_prompt()}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": message})
    response_message = run_chat(messages)
    return response_message.content or ""


def gather_blocking_issues() -> List[str]:
    issues: List[str] = []
    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY is missing.")
    for required in (SUMMARY_PATH, PROFILE_PATH):
        if not required.exists():
            issues.append(f"Required file missing: {required}")
    return issues


chat_interface = gr.ChatInterface(
    fn=chat,
    title="Ebube Imoh | Career Companion",
    description="Chat with Ebube's AI assistant to learn more about his background and get in touch.",
)


if __name__ == "__main__":
    blocking = gather_blocking_issues()
    if blocking:
        for issue in blocking:
            logger.error(issue)
        raise SystemExit("Cannot launch chat interface due to missing configuration.")

    if not (PUSHOVER_USER and PUSHOVER_TOKEN):
        logger.warning("Pushover credentials are not configured; notifications will be skipped.")

    chat_interface.launch()
