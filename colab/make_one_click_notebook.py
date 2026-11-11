"""Build a self-contained notebook with exactly one executable code cell.

Run from bbh/: python3 colab/make_one_click_notebook.py
"""
import base64
import io
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = [*sorted((root / "py" / "bbh").glob("*.py")),
           root / "contract" / "spec.json", root / "contract" / "goal_blocks.json"]
stream = io.BytesIO()
with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zipped:
    for path in sources:
        zipped.write(path, path.relative_to(root).as_posix())
blob = base64.b64encode(stream.getvalue()).decode("ascii")
literal = "(\n" + "\n".join(repr(blob[i:i + 110]).join(("    ", ""))
                           for i in range(0, len(blob), 110)) + "\n)"
runner = (root / "colab" / "one_click_runner.py").read_text()
assert runner.count("source_archive = __ARCHIVE_PLACEHOLDER__") == 1
runner = runner.replace("source_archive = __ARCHIVE_PLACEHOLDER__", "source_archive = " + literal)

guide = """# BBH — one-click Google Colab training

**There is only one code cell.** Set your Colab runtime to an L4 or T4 GPU and
click Run once. It installs missing packages, downloads real OpenAI VPT gameplay,
converts and checks it, trains, verifies ONNX, then downloads the model ZIP.
**No GitHub clone or separate project ZIP upload required.**

By default `SMOKE_TEST = True` executes a short **30-step engineering test**.
It is **not a useful trained Minecraft agent**. Once it succeeds, change that
single setting at the top of the code cell to `False` and run the same cell for
150 recordings and 20,000 steps; this may take many hours.

Checkpoints and a persistent copy of the training shards live in
`MyDrive/bbh_work/` (they require several GB of free Drive space for a full
run). If Colab disconnects, rerun the same cell: the saved shards are restored
to fast local storage and training resumes. Google may ask to authorize Drive.
Colab session limits still apply. The Fabric mod has not been compiled or
tested in Minecraft.
"""
notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU",
                         "colab": {"name": "BBH_Train_One_Click.ipynb", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": guide.splitlines(True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [], "source": runner.splitlines(True)},
            ]}
out = root / "colab" / "BBH_Train_One_Click.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out} ({out.stat().st_size // 1024} KiB, exactly one code cell)")
