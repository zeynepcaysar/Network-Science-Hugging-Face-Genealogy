"""Shared helpers of the Phase 4 tests: the scripts in data_supplement/ loaded by file, a fake Hub API and builders for
Hub models and errors. No fixtures and no network: every test passes a FakeApi, never a real HfApi.

    from outside_fixture import FakeApi, load_script, model, not_found_error

    fetch = load_script("fetch_outside")                    # data_supplement/fetch_outside.py as a module, loaded once
    api = FakeApi([model("org/name", createdAt="2024-08-03T10:00:00.000Z", downloads=5)])
"""
import importlib.util
from pathlib import Path

import httpx2
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import ModelInfo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "data_supplement"
_LOADED = {}


def load_script(name):
    """data_supplement/NAME.py as a module (loaded once per session, by file, so no package or sys.path change is needed)."""
    if name not in _LOADED:
        path = SCRIPTS / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"data_supplement_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[name] = module
    return _LOADED[name]


def model(id, **api_json):
    """A huggingface_hub ModelInfo built from API-style keyword names (createdAt, safetensors, downloadsAllTime, ...)."""
    return ModelInfo(id=id, **api_json)


def _response(code, repo_id="org/name", headers=None):
    request = httpx2.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
    return httpx2.Response(code, headers=headers or {}, request=request)


def not_found_error(repo_id, code=401):
    """What the Hub client raises for a missing repo: HTTP 401 anonymously, 404 with a token."""
    return RepositoryNotFoundError("Repository Not Found", response=_response(code, repo_id))


class FakeApi:
    """Stands in for HfApi: list_models yields the listing, model_info calls info(repo_id) and returns or raises the result.

    list_calls records the keyword arguments of every list_models call; info_calls records (repo_id, kwargs) of every
    model_info call. Without an info callable every model_info raises the 401 not-found error.
    """

    def __init__(self, listing, info=None):
        self.listing = list(listing)
        self.info = info
        self.list_calls = []
        self.info_calls = []

    def list_models(self, **kwargs):
        self.list_calls.append(kwargs)
        limit = kwargs.get("limit")
        for i, m in enumerate(self.listing):
            if limit is not None and i >= limit:
                return
            yield m

    def model_info(self, repo_id, **kwargs):
        self.info_calls.append((repo_id, kwargs))
        if self.info is None:
            raise not_found_error(repo_id)
        result = self.info(repo_id)
        if isinstance(result, BaseException):
            raise result
        return result
