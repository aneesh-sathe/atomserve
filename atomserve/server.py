# OpenAI-compatible HTTP API, so curl, the `openai` client or Open WebUI can use the engine.
#
#     GET  /v1/models             the one model being served
#     POST /v1/chat/completions   messages -> reply   (streaming or not)
#     POST /v1/completions        raw prompt -> continuation
#     GET  /health                200 once the server is up
#
# Every request runs on its own handler thread (stdlib ThreadingHTTPServer)
# and is just another client of the engine, exactly like the REPL: render and
# tokenize, submit, then turn the stream of token ids back into text.
# Streaming responses are Server-Sent Events: one `data: {json}` line per
# chunk, ending with `data: [DONE]`.
#
# Two things the engine deliberately doesn't know about are handled here:
# stop strings (matched on the decoded text, then the request is aborted) and
# detecting a client that went away mid-stream (also an abort).
#
# Extensions beyond OpenAI's API, following vLLM: `top_k`, `ignore_eos`, and
# `chat_template_kwargs: {"enable_thinking": false}` to turn off Qwen3's thinking.

import itertools
import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from atomserve.engine import Engine, RequestHandle
from atomserve.sequence import NON_THINKING_PARAMS, THINKING_PARAMS, SamplingParams
from atomserve.tokenizer import ChatTokenizer, IncrementalDetokenizer


class BadRequest(ValueError):
    """The client sent something we can't serve: answered with HTTP 400."""


class StopChecker:
    """Finds stop strings in streamed text without ever emitting part of one.

    Text that could still turn out to be the start of a stop string is held
    back until the next delta decides it: with stop "###", the delta "a##" emits
    "a" and holds "##".
    """

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.held = ""

    def feed(self, delta: str) -> tuple[str, bool]:
        """Returns (text safe to emit, whether a stop string was hit)."""
        text = self.held + delta
        hits = [i for i in (text.find(s) for s in self.stops) if i >= 0]
        if hits:
            self.held = ""
            return text[: min(hits)], True
        keep = max(
            (k for s in self.stops for k in range(1, len(s)) if text.endswith(s[:k])), default=0
        )
        self.held = text[len(text) - keep :] if keep else ""
        return text[: len(text) - keep], False

    def flush(self) -> str:
        """Release what is held back once the stream has ended without a stop."""
        held, self.held = self.held, ""
        return held


class Generation:
    """One request's text stream: token ids -> text deltas, with stop strings applied."""

    def __init__(self, handle: RequestHandle, tokenizer: ChatTokenizer, stops: list[str]):
        self.handle = handle
        self.detok = IncrementalDetokenizer(tokenizer)
        self.stops = StopChecker(stops)
        self.finish_reason: str | None = None

    def texts(self) -> Iterator[str]:
        """Yield text deltas. Closing the iterator early (client gone) aborts the request."""
        try:
            for token in self.handle:
                text, stopped = self.stops.feed(self.detok.push(token))
                if text:
                    yield text
                if stopped:
                    self.handle.abort()
                    for _ in self.handle:  # drain until the engine confirms
                        pass
                    self.finish_reason = "stop"
                    return
            if tail := self.stops.feed(self.detok.flush())[0] + self.stops.flush():
                yield tail
            self.finish_reason = self.handle.finish_reason
        finally:
            if self.handle.finished_at is None and self.finish_reason is None:
                self.handle.abort()

    @property
    def usage(self) -> dict:
        prompt, completion = self.handle.num_prompt_tokens, self.handle.num_output_tokens
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


def message_text(message: dict) -> str:
    """A message's content as plain text; OpenAI also allows a list of text parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") == "text" for p in content
    ):
        return "".join(p.get("text", "") for p in content)
    if content is None and message.get("role") == "assistant":
        return ""
    raise BadRequest("message content must be a string or a list of text parts")


def sampling_params(body: dict, defaults: SamplingParams, room: int) -> SamplingParams:
    """Map OpenAI request fields onto SamplingParams; unset fields keep `defaults`.

    `room` is how many tokens fit after the prompt: the default and the cap
    for max_tokens.
    """
    fields: dict = {"max_tokens": room}
    types = {"temperature": float, "top_p": float, "top_k": int, "presence_penalty": float}
    types |= {"seed": int, "max_tokens": int, "max_completion_tokens": int}
    for key, kind in types.items():
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadRequest(f"{key} must be a number")
        fields["max_tokens" if key == "max_completion_tokens" else key] = kind(value)
    if fields.get("temperature", 0) < 0 or not 0 < fields.get("top_p", 1) <= 1:
        raise BadRequest("temperature must be >= 0 and top_p in (0, 1]")
    if fields["max_tokens"] < 1:
        raise BadRequest("max_tokens must be at least 1")
    fields["max_tokens"] = min(fields["max_tokens"], room)
    fields["ignore_eos"] = bool(body.get("ignore_eos", False))
    return replace(defaults, **fields)


def stop_strings(body: dict) -> list[str]:
    stop = body.get("stop") or []
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise BadRequest("stop must be a string or a list of strings")
    return stops


class OpenAIServer:
    """Serves the engine over HTTP on a background thread."""

    def __init__(
        self,
        engine: Engine,
        tokenizer: ChatTokenizer,
        model_name: str,
        host: str = "127.0.0.1",
        port: int = 11435,
        log_requests: bool = False,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.log_requests = log_requests
        self._ids = itertools.count(1)
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.app = self  # the handler reaches the server's state through this
        self.port = self.httpd.server_address[1]  # the real one, if port was 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="atomserve-http", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def submit(self, prompt_ids: list[int], body: dict, defaults: SamplingParams) -> Generation:
        """Validate and submit one request; returns its text stream."""
        room = self.engine.scheduler.config.max_model_len - len(prompt_ids)
        if room < 1:
            raise BadRequest(
                f"prompt is {len(prompt_ids)} tokens; the limit is "
                f"{self.engine.scheduler.config.max_model_len - 1}"
            )
        if body.get("n", 1) != 1:
            raise BadRequest("only n=1 is supported")
        params = sampling_params(body, defaults, room)
        stops = stop_strings(body)
        try:
            handle = self.engine.submit(prompt_ids, params)
        except ValueError as error:
            raise BadRequest(str(error)) from error
        return Generation(handle, self.tokenizer, stops)

    def next_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids)}"


class _Handler(BaseHTTPRequestHandler):
    """Routes one HTTP request. A new instance per request, on its own thread."""

    server_version = "atomserve"

    @property
    def app(self) -> OpenAIServer:
        return self.server.app

    def log_message(self, format: str, *args) -> None:
        # Silent by default: the REPL owns the terminal.
        if self.app.log_requests:
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            model = {"id": self.app.model_name, "object": "model", "owned_by": "atomserve"}
            self._json(200, {"object": "list", "data": [model]})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._error(404, f"no route GET {self.path}", "not_found_error")

    def do_POST(self) -> None:
        routes = {"/v1/chat/completions": self._chat, "/v1/completions": self._completion}
        if self.path not in routes:
            self._error(404, f"no route POST {self.path}", "not_found_error")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            routes[self.path](body)
        except json.JSONDecodeError as error:
            self._error(400, f"invalid JSON: {error}", "invalid_request_error")
        except BadRequest as error:
            self._error(400, str(error), "invalid_request_error")
        except BrokenPipeError, ConnectionResetError:
            pass  # client went away; Generation already aborted its request
        except Exception as error:  # report rather than drop the connection
            self._error(500, f"{type(error).__name__}: {error}", "server_error")

    def _chat(self, body: dict) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("messages must be a non-empty list")
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
                raise BadRequest("each message needs a role of system, user or assistant")
        messages = [{"role": m["role"], "content": message_text(m)} for m in messages]
        kwargs = body.get("chat_template_kwargs") or {}
        thinking = bool(kwargs.get("enable_thinking", True))
        prompt = self.app.tokenizer.encode_chat(messages, enable_thinking=thinking)
        defaults = THINKING_PARAMS if thinking else NON_THINKING_PARAMS
        generation = self.app.submit(prompt, body, defaults)
        request_id = self.app.next_id("chatcmpl")

        def chunk(delta: dict, finish_reason: str | None) -> dict:
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            return self._envelope(request_id, "chat.completion.chunk", [choice])

        if body.get("stream"):
            self._start_stream()
            self._event(chunk({"role": "assistant", "content": ""}, None))
            with closing(generation.texts()) as texts:
                for text in texts:
                    self._event(chunk({"content": text}, None))
            self._event(chunk({}, generation.finish_reason))
            self._end_stream(request_id, "chat.completion.chunk", generation, body)
        else:
            text = "".join(generation.texts())
            message = {"role": "assistant", "content": text}
            choice = {"index": 0, "message": message, "finish_reason": generation.finish_reason}
            response = self._envelope(request_id, "chat.completion", [choice])
            self._json(200, {**response, "usage": generation.usage})

    def _completion(self, body: dict) -> None:
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            raise BadRequest("prompt must be a non-empty string")
        generation = self.app.submit(self.app.tokenizer.encode(prompt), body, THINKING_PARAMS)
        request_id = self.app.next_id("cmpl")

        def choice(text: str, finish_reason: str | None) -> dict:
            return {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}

        if body.get("stream"):
            self._start_stream()
            with closing(generation.texts()) as texts:
                for text in texts:
                    self._event(self._envelope(request_id, "text_completion", [choice(text, None)]))
            final = choice("", generation.finish_reason)
            self._event(self._envelope(request_id, "text_completion", [final]))
            self._end_stream(request_id, "text_completion", generation, body)
        else:
            text = "".join(generation.texts())
            response = self._envelope(
                request_id, "text_completion", [choice(text, generation.finish_reason)]
            )
            self._json(200, {**response, "usage": generation.usage})

    # ---- response helpers ---------------------------------------------------------

    def _envelope(self, request_id: str, kind: str, choices: list[dict]) -> dict:
        return {
            "id": request_id,
            "object": kind,
            "created": int(time.time()),
            "model": self.app.model_name,
            "choices": choices,
        }

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str) -> None:
        self._json(status, {"error": {"message": message, "type": kind, "code": status}})

    def _start_stream(self) -> None:
        # HTTP/1.0 semantics: no length header, the body ends when the connection closes.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _event(self, payload: dict | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def _end_stream(self, request_id: str, kind: str, generation: Generation, body: dict) -> None:
        if (body.get("stream_options") or {}).get("include_usage"):
            self._event({**self._envelope(request_id, kind, []), "usage": generation.usage})
        self._event("[DONE]")
