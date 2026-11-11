"""Downloader regression checks. No network or video dependencies.

Run: PYTHONPATH=py python3 tests/test_download.py
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from bbh import data


def main():
    base = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
    rel = "data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4"
    mp4, jl = data.segment_urls(base, rel)
    assert mp4 == base + rel, mp4
    assert jl == base + rel[:-4] + ".jsonl", jl
    assert not jl.endswith(".mp4.jsonl")
    assert data.seg_name(rel).endswith("20220421-092639")
    assert data.segment_urls("https://example.com/data/", "8.0/example") == (
        "https://example.com/data/8.0/example.mp4",
        "https://example.com/data/8.0/example.jsonl")

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        (cache / "sample.json").write_text(json.dumps({"basedir": base,
               "relpaths": ["data/10.0/missing.mp4", "data/10.0/missing2.mp4",
                            "data/10.0/good1.mp4", "data/10.0/good2.mp4"]}))
        assert data.fetch_index("sample.json", cache) == (
            base, ["data/10.0/missing.mp4", "data/10.0/missing2.mp4",
                   "data/10.0/good1.mp4", "data/10.0/good2.mp4"])
        candidates = [(base, rel) for rel in ["data/10.0/missing.mp4",
                      "data/10.0/missing2.mp4", "data/10.0/good1.mp4", "data/10.0/good2.mp4"]]

        def fake_download(_base, candidate, _raw):
            if "missing" in candidate:
                return None
            return (cache / "vid.mp4", cache / "action.jsonl")

        with patch.object(data, "download_segment", side_effect=fake_download) as mocked:
            got = data.download_first_valid(candidates, cache, 2, workers=2, max_attempts=4,
                                            log=lambda *_: None)
            assert {r for _, r in got} == {"data/10.0/good1.mp4", "data/10.0/good2.mp4"}, got
            assert mocked.call_count == 4
        with patch.object(data, "download_segment", return_value=None):
            try:
                data.download_first_valid(candidates, cache, 1, max_attempts=2, log=lambda *_: None)
            except RuntimeError as error:
                assert "Only 0/1" in str(error)
            else:
                raise AssertionError("Zero downloads must stop before training")
    print("downloader tests passed")


if __name__ == "__main__":
    main()
