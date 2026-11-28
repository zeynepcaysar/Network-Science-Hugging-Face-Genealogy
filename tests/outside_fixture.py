"""Shared helpers of the Phase 4 tests: the scripts in data_supplement/ loaded by file, a fake Hub API and builders for
Hub models and errors. No fixtures and no network: every test passes a FakeApi, never a real HfApi.

    from outside_fixture import FakeApi, load_script, model, not_found_error

    fetch = load_script("fetch_outside")                    # data_supplement/fetch_outside.py as a module, loaded once
    api = FakeApi([model("org/name", createdAt="2024-08-03T10:00:00.000Z", downloads=5)])
    raw = make_raw(tmp_path, ids)                           # frozen raw_hub_fetch.parquet of the data_mini ids (load.DATA must point at the copy)
"""
import importlib.util
import json
from pathlib import Path

import httpx2
import pandas as pd
from huggingface_hub.errors import DisabledRepoError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import ModelInfo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "data_supplement"
RAW_MINI = ROOT / "tests" / "fixtures" / "outside" / "raw_mini.jsonl"  # hand-written raw fetch of the data_mini ids
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


def http_error(code, headers=None, repo_id="org/name"):
    """An HfHubHTTPError on an HTTP response with this status code and these headers."""
    return HfHubHTTPError(str(code), response=_response(code, repo_id, headers))


def gated_error(repo_id):
    return GatedRepoError("Gated repo", response=_response(401, repo_id))


def disabled_error(repo_id):
    return DisabledRepoError("Disabled repo", response=_response(403, repo_id))


def sequence_info(outcomes):
    """An info callable for FakeApi: outcomes maps a repo id to a list of results (exceptions or ModelInfo), used in order.

    The last result of a list repeats forever; an id that is not in outcomes raises the 401 not-found error.
    """
    pending = {k: list(v) for k, v in outcomes.items()}

    def info(repo_id):
        if repo_id not in pending:
            return not_found_error(repo_id)
        queue = pending[repo_id]
        return queue.pop(0) if len(queue) > 1 else queue[0]
    return info


class Recorder:
    """A sleep replacement: records the requested waits and returns at once."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


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


# ---------------------------------------------------------------- raw fetch and supplement files
def raw_rows():
    """The rows of tests/fixtures/outside/raw_mini.jsonl as dicts (keys = fetch_outside.RAW_COLUMNS), one per data_mini id."""
    return [json.loads(line) for line in RAW_MINI.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_raw(tmp_path, ids, rows=None, test_run=False):
    """Freeze rows (default: raw_mini.jsonl) in the real raw format and return the path of the parquet.

    The rows are written as a listing work file to <tmp>/out/raw_list.jsonl and frozen with fetch_outside.freeze_raw, which
    records load.DATA's nodes.csv and MANIFEST.txt, so the caller points load.DATA at a data copy with a MANIFEST first.
    """
    fetch = load_script("fetch_outside")
    out = Path(tmp_path) / "out"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / fetch.LIST_NAME, "w", encoding="utf-8", newline="\n") as f:
        for row in (raw_rows() if rows is None else rows):
            f.write(json.dumps(row) + "\n")
    fetch.freeze_raw(out, ids, test_run=test_run)
    return out / fetch.RAW_NAME


def write_supplement_files(tmp_path, rows):
    """Write date_supplement.csv and its sidecar into <tmp>/supp from rows (dicts with at least id, created_at and
    date_is_placeholder; the other columns get defaults); an empty list gives a header-only supplement. Returns the CSV path."""
    build = load_script("build_date_supplement")
    base = {"status": "ok", "resolved_id": "", "created_at": "", "date_is_placeholder": False, "shared_repo": False,
            "source": "list", "fetched_at": "2026-11-28T09:00:00+00:00", "v1_reason": "no_metadata", "downloads_30d": pd.NA,
            "downloads_all_time": pd.NA, "likes": pd.NA, "note": ""}
    full = [{**base, "resolved_id": r["id"], **r} for r in rows]
    frame = pd.DataFrame(full, columns=list(build.SUPPLEMENT_COLUMNS))
    for col in ("downloads_30d", "downloads_all_time", "likes"):
        frame[col] = pd.array(frame[col], dtype="Int64")
    out = Path(tmp_path) / "supp"
    out.mkdir(parents=True, exist_ok=True)
    build.write_supplement(frame, {"script": "tests/outside_fixture.py"}, out)
    return out / build.CSV_NAME
