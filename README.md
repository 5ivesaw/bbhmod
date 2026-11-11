# bbh: a learned Minecraft motor policy

Train in Google Colab on real human gameplay (OpenAI VPT contractor data), export one small ONNX file,
and run it locally inside a Fabric mod. No cloud and no internet needed once installed. Baritone is optional:
the policy can hand a mining task to it when it gets stuck.

```
Colab:  VPT segments -> convert (64x64 frames, keys, camera, hotbar, inventory, goals) -> train -> policy.onnx
Mod:    framebuffer -> shrink -> ONNX (1 CPU thread) -> keys / camera / hotbar -> player   (+ Baritone handoff)
```

## What it learns

A small recurrent network (about 1M parameters) sees a 64x64 picture of the screen, your hotbar slot, camera angles,
and inventory counts, plus an optional goal item. Each 0.1 s it outputs which keys are down, a camera turn, and a
hotbar change. Goals come from hindsight: in the data, the goal for a moment is the item the player gained next.
So `/bbh goal log` means "behave like players who were about to collect logs."

## Files

| Path | What |
|---|---|
| `colab/BBH_Train_One_Click.ipynb` | **Recommended:** self-contained Colab notebook with one Run cell, no GitHub or ZIP upload |
| `colab/BBH_Train.ipynb` | Original multi-cell notebook, kept only for reference (its downloader was broken) |
| `colab/one_click_runner.py` and `make_one_click_notebook.py` | Source and generator for the single-cell notebook |
| `py/bbh/` | `data.py` download and convert, `model.py`, `train.py`, `export.py`, `encode.py` (reference math) |
| `contract/spec.json` | The single shared definition of inputs and outputs |
| `contract/goal_blocks.json` | Goal item to blocks Baritone should mine |
| `mod/` | Fabric mod (Gradle project). `core/` has no Minecraft dependency |
| `tests/` | Converter test and Python-to-Java parity fixtures |

## Step 1: train (Colab)

1. Open **`colab/BBH_Train_One_Click.ipynb`** in Colab; select L4 or T4 GPU.
2. Click the **single code cell's Run button**. It installs dependencies, reads real VPT indexes, replaces missing
   recordings, builds vocabulary, converts data, sanity-checks frames and actions, trains, exports ONNX, and downloads
   `bbh_agent_model.zip` automatically. No repo clone or manual project upload.
3. Leave `SMOKE_TEST = True` for the first pass (6 segments, **30** training steps). This is a pipeline test, not a
   useful trained model. Check its frame preview and summary for plausible gameplay, then switch it to `False`
   (150 segments, 20,000 steps) and run the same cell again.
4. Drive retains the training checkpoint **and a cache of converted shards**. A disconnected session can rerun the
   same cell to rebuild the local fast copy and resume. This uses Drive space (several GB on a full run).
   Google may prompt you to grant Drive access. Colab session limits still apply.

## Step 2: build the mod

```
cd mod
./gradlew parityTest     # pure Java check against the Python math, no Minecraft needed
./gradlew build          # needs internet for Fabric/Minecraft; jar ends up in build/libs
```

Put the unzipped model in `mod/model/` before building to bundle it in the jar, or copy it to
`.minecraft/config/bbh/` (files there win). Needs Java 25, Fabric Loader, Fabric API, and optionally a Baritone
Fabric jar from [Baritone releases](https://github.com/cabaletta/baritone/releases) for your Minecraft version.

## Step 3: first run (do these in order)

1. `/bbh dumpframe`, then open `config/bbh/frame.png`. Does it look like your game with correct colors?
   If reds and blues are swapped: `/bbh channels abgr`.
2. `/bbh goal log` then `/bbh start`. `/bbh status` shows policy time per step, frame capture cost, and what
   share of ticks the learned model versus Baritone controlled.
3. Open any menu, or alt-tab, and it pauses and releases every key. `/bbh stop` stops it.

Commands: `start`, `stop`, `status`, `goal <item|none>`, `mode <learned|hybrid|baritone>`, `greedy <true|false>`,
`temp <keys> <camera>`, `channels <argb|abgr>`, `dumpframe`, `goto <x> <y> <z>`.

## What was actually verified (and what was not)

Previously reported for the original project (not independently rerun here):
- The converter, run on a synthetic segment written in the documented VPT format: frame/action alignment, camera
  bins, key labels, GUI masking, hotbar changes, hindsight goals. The test caught and I fixed a vocabulary bug.
- The Java encoders (image shrink, camera bins, item names, state vector) against fixtures from the Python
  reference. Image shrink matches within 1 brightness level; everything else matches exactly.

Verified for this update:
- The single-cell notebook parses and bundles the Python source directly.
- Actual VPT 10.x index entries are formatted `data/10.0/<name>.mp4`; the downloader strips `.mp4` before making
  the `.jsonl` URL. Regression tests cover both current and older documented formats and the missing-file fallback.

NOT verified, so expect to fix things:
- **Nothing was trained.** PyTorch is not available in this sandbox. `model.py`, `train.py` and `export.py` have only
  been reviewed by reading, not run. The smoke test in step 1 exists to catch exactly that.
- **Real VPT videos were not decoded in this environment.** The live index structure and a matching JSONL HTTP
  response were verified, but the full Colab pipeline is not yet verified end to end. The smoke run is the check.
- **The mod was never compiled against Minecraft 26.3.** `MinecraftBridge.java`, `FrameGrabber.java` and `Main.java`
  call Minecraft internals from my knowledge of recent versions (`Entity.turn`, `Screenshot.takeScreenshot`,
  `NativeImage.getPixel`, `Inventory.getSelectedSlot`, key mappings). Any of these may be renamed in 26.3. They are
  isolated in those three files so fixing a compile error is small. Gradle versions came from Fabric's 26.3 post and
  a Modrinth listing; if Gradle can't resolve one, take the current values from the Fabric template generator.
- **Camera direction is unconfirmed.** I assumed VPT's yaw/pitch signs match what `Entity.turn` expects. Watch the
  first seconds of `/bbh start`: if it looks away from where it should, flip the sign in `MinecraftBridge.turn`.
- **Speed on an i5-6200U with HD 520 is unmeasured.** The network itself is tiny. The unknown is the screen read-back
  on integrated graphics; `/bbh status` prints it. If it hurts, lower the model rate by raising `stride` in
  `spec.json` and retraining.
- **Baritone handoff** uses its public API via reflection and has not run against a real Baritone build.

Honest limits of the approach: this is behaviour cloning on contractor gameplay at 64x64 resolution. Expect it to
walk, look around, chop and mine in a roughly human way; do not expect it to craft reliably or navigate complex
terrain on its own. That is what the Baritone handoff is for.
