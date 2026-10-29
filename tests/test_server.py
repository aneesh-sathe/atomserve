# HTTP API tests. Fast: the pure pieces (stop strings, request parsing). Slow: a real
# server on an ephemeral port in front of the real model, driven with raw HTTP
# (to check the wire format) and with the official `openai` client.

import http.client
import json
import socket
import time

import pytest

from atomserve.config import EngineConfig
from atomserve.engine import create_engine
from atomserve.sequence import THINKING_PARAMS
from atomserve.server import BadRequest, OpenAIServer, StopChecker, message_text, sampling_params
from atomserve.tokenizer import ChatTokenizer


def run_stops(stops: list[str], deltas: list[str]) -> tuple[str, bool]:
    checker, emitted = StopChecker(stops), []
    for delta in deltas:
        text, stopped = checker.feed(delta)
        emitted.append(text)
        if stopped:
            return "".join(emitted), True
    return "".join(emitted) + checker.flush(), False


def test_stop_string_split_across_deltas_is_never_emitted():
    assert run_stops(["###"], ["a#", "#", "#b"]) == ("a", True)


def test_held_back_text_is_released_when_it_is_not_a_stop():
    assert run_stops(["###"], ["a#", "#", "b"]) == ("a##b", False)
    assert run_stops(["###"], ["a##"]) == ("a##", False)  # flushed at the end


def test_earliest_of_several_stop_strings_wins():
    assert run_stops(["world", "lo"], ["hello world"]) == ("hel", True)


def test_sampling_params_defaults_caps_and_validation():
    p = sampling_params({}, THINKING_PARAMS, room=100)
    assert (p.temperature, p.top_p, p.top_k, p.max_tokens) == (0.6, 0.95, 20, 100)
    p = sampling_params({"max_completion_tokens": 500, "temperature": 0}, THINKING_PARAMS, 100)
    assert (p.max_tokens, p.temperature) == (100, 0.0)  # capped by the room left
    for bad in ({"temperature": "hot"}, {"top_p": 0}, {"max_tokens": 0}, {"seed": True}):
        with pytest.raises(BadRequest):
            sampling_params(bad, THINKING_PARAMS, 100)


def test_message_content_may_be_text_parts():
    parts = [{"type": "text", "text": "Hello, "}, {"type": "text", "text": "world"}]
    assert message_text({"role": "user", "content": parts}) == "Hello, world"
    with pytest.raises(BadRequest):
        message_text({"role": "user", "content": [{"type": "image_url"}]})


@pytest.fixture(scope="module")
def server(model_dir, weights_bf16, config):
    tokenizer = ChatTokenizer(model_dir)
    engine_config = EngineConfig(num_blocks=512, max_model_len=8192)
    engine = create_engine(weights_bf16, config, engine_config, tokenizer.eos_token_ids)
    engine.start()
    server = OpenAIServer(engine, tokenizer, "test-model", port=0)
    server.start()
    yield server
    server.shutdown()
    engine.shutdown()


def post(server, path: str, body) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=120)
    conn.request("POST", path, body if isinstance(body, str) else json.dumps(body))
    response = conn.getresponse()
    return response.status, response.read()


NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}


@pytest.mark.slow
def test_chat_completion(server):
    body = {"messages": [{"role": "user", "content": "What is 2+2? Answer with a digit."}]}
    status, raw = post(server, "/v1/chat/completions", {**body, **NO_THINK})
    reply = json.loads(raw)
    assert status == 200
    assert reply["object"] == "chat.completion"
    assert "4" in reply["choices"][0]["message"]["content"]
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["usage"]["total_tokens"] == (
        reply["usage"]["prompt_tokens"] + reply["usage"]["completion_tokens"]
    )


@pytest.mark.slow
def test_streaming_wire_format(server):
    """Every event is `data: <json>`; role first, a finish chunk, usage, then [DONE]."""
    body = {
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": True,
        "stream_options": {"include_usage": True},
        **NO_THINK,
    }
    status, raw = post(server, "/v1/chat/completions", body)
    assert status == 200
    events = [e for e in raw.decode().split("\n\n") if e]
    assert all(e.startswith("data: ") for e in events)
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[6:]) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[:-1])
    assert "hello" in text.lower()


@pytest.mark.slow
def test_completion_stops_at_stop_string(server):
    body = {"prompt": "1, 2, 3,", "max_tokens": 20, "temperature": 0, "stop": [" 6"]}
    status, raw = post(server, "/v1/completions", body)
    choice = json.loads(raw)["choices"][0]
    assert status == 200
    assert choice["text"] == " 4, 5,"
    assert choice["finish_reason"] == "stop"


@pytest.mark.slow
def test_errors_are_openai_shaped(server):
    for body in ("not json", {"messages": []}, {"prompt": "x", "temperature": -1}):
        path = "/v1/completions" if "prompt" in str(body) else "/v1/chat/completions"
        status, raw = post(server, path, body)
        assert status == 400
        assert json.loads(raw)["error"]["type"] == "invalid_request_error"
    status, _ = post(server, "/v1/nope", {})
    assert status == 404


@pytest.mark.slow
def test_client_disconnect_aborts_the_request(server):
    """Closing the socket mid-stream frees the request's KV blocks within a moment."""
    body = {"prompt": "Once upon a time", "max_tokens": 4000, "ignore_eos": True, "stream": True}
    sock = socket.create_connection(("127.0.0.1", server.port))
    payload = json.dumps(body).encode()
    sock.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: "
        + str(len(payload)).encode()
        + b"\r\n\r\n"
        + payload
    )
    received = b""
    while b"data: " not in received:  # headers and the first event may arrive separately
        received += sock.recv(4096)
    assert server.engine.stats()["running"] == 1
    sock.close()
    deadline = time.time() + 10
    while server.engine.stats()["running"] and time.time() < deadline:
        time.sleep(0.05)
    assert server.engine.stats()["running"] == 0


@pytest.mark.slow
def test_openai_client_streams(server):
    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1", api_key="unused")
    assert [m.id for m in client.models.list()] == ["test-model"]
    stream = client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "Name a color. One word."}],
        stream=True,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text.strip()
