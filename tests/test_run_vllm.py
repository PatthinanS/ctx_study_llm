import json
from unittest.mock import MagicMock, patch

from src.run import _guided_json_kwargs, call_llm_once, call_llm_with_retry


def _vllm_cfg(base_url="http://fake-vllm:8000"):
    return {"model": "fake-model", "backend": "vllm", "vllm_base_url": base_url}


def _ollama_cfg():
    # No "backend" key -- must resolve to "ollama" via cfg.get("backend", "ollama").
    return {"model": "fake-model"}


def _mock_response(status_code=200, body=None, raises=None):
    resp = MagicMock(status_code=status_code)
    if raises is not None:
        resp.raise_for_status = MagicMock(side_effect=raises)
    else:
        resp.raise_for_status = MagicMock()
    resp.json.return_value = body or {}
    return resp


def test_guided_json_kwargs_shape():
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    assert _guided_json_kwargs(schema) == {"guided_json": schema}


def test_call_llm_with_retry_vllm_succeeds_on_second_attempt():
    good_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"label": "hap", "vad": {"v": 4.0, "a": 3.0, "d": 3.0}})
                }
            }
        ]
    }
    mock_post = MagicMock(side_effect=[RuntimeError("boom"), _mock_response(body=good_body)])
    with patch("requests.post", mock_post):
        label, vad, latency_ms, raw = call_llm_with_retry(
            "fake-model", "sys", "user", {"temperature": 0, "seed": 42}, {}, _vllm_cfg()
        )
    assert label == "hap"
    assert vad == {"v": 4.0, "a": 3.0, "d": 3.0}
    assert mock_post.call_count == 2


def test_call_llm_with_retry_vllm_fails_gracefully_after_two_attempts():
    mock_post = MagicMock(side_effect=[RuntimeError("boom"), RuntimeError("boom again")])
    with patch("requests.post", mock_post):
        label, vad, latency_ms, raw = call_llm_with_retry(
            "fake-model", "sys", "user", {"temperature": 0, "seed": 42}, {}, _vllm_cfg()
        )
    assert label is None
    assert vad == {"v": None, "a": None, "d": None}
    assert mock_post.call_count == 2


def test_call_llm_once_vllm_posts_to_chat_completions_with_guided_json():
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    body = {"choices": [{"message": {"content": json.dumps({"label": "neu"})}}]}
    mock_post = MagicMock(return_value=_mock_response(body=body))
    with patch("requests.post", mock_post):
        parsed, raw, latency_ms = call_llm_once(
            "fake-model", "sys", "user", {"temperature": 0, "seed": 42}, schema,
            _vllm_cfg(base_url="http://fake-vllm:8000"),
        )
    assert parsed == {"label": "neu"}
    called_url = mock_post.call_args.args[0]
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_url == "http://fake-vllm:8000/v1/chat/completions"
    assert called_payload["guided_json"] == schema
    assert called_payload["temperature"] == 0
    assert called_payload["seed"] == 42


def test_call_llm_with_retry_ollama_backend_unaffected_by_vllm_mock():
    """A cfg with no 'backend' key (implicit 'ollama') must route to
    call_ollama_with_retry and never touch requests.post -- proves the
    dispatcher's default path is completely unaffected by the vLLM code
    existing at all, even if requests.post is mocked to blow up.
    """
    good_ollama_response = {
        "message": {
            "content": json.dumps({"label": "neu", "vad": {"v": 3.0, "a": 3.0, "d": 3.0}})
        }
    }
    mock_post = MagicMock(side_effect=AssertionError("requests.post should not be called"))
    mock_chat = MagicMock(return_value=good_ollama_response)
    with patch("requests.post", mock_post), patch("ollama.chat", mock_chat):
        label, vad, latency_ms, raw = call_llm_with_retry(
            "fake-model", "sys", "user", {"temperature": 0, "seed": 42}, {}, _ollama_cfg()
        )
    assert label == "neu"
    assert mock_chat.call_count == 1
    mock_post.assert_not_called()
