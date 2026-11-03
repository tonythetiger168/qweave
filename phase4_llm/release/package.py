"""
Build (and, only with approval, upload) a DiracLM release folder
================================================================
  # build locally: weights + diraclm_config.json + README.md + eval/ + MANIFEST
  python -m phase4_llm.release.package build \
      --recipe phase4_llm/release/recipes/diraclm-4b.json \
      --artifacts /workspace/ql8 --weights /workspace/ql8/ql7_4b/final \
      --base-eval /workspace/ql8/eval/qwen3-4b-base \
      --model-eval /workspace/ql8/eval/diraclm-4b --out /workspace/ql8/release

  # upload: PRIVATE repo by default; refuses without --approved-by, and
  # refuses --public unless the release gate passed
  python -m phase4_llm.release.package push --folder /workspace/ql8/release \
      --approved-by "Tony, thread message 2026-11-xx" [--public]

`build` never touches the network. `push` is the only function that
publishes anything.
"""

import argparse
import hashlib
import json
import os
import shutil
from typing import Dict, Optional

from phase4_llm.release import model_card, reasoning_eval
from phase4_llm.release.loader import CONFIG_NAME
from phase4_llm.release.recipe import load_recipe, release_gate, resolve

WEIGHT_FILES = (".safetensors", ".json", ".txt", ".model", ".tiktoken", ".jinja", ".py")
SKIP = {"README.md", CONFIG_NAME, "MANIFEST.json", "training_args.bin"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _public_eval_copy(src_dir: str, dst_dir: str):
    """Copy per-item eval files; GPQA rows keep ids/preds/scores only."""
    os.makedirs(dst_dir, exist_ok=True)
    for fn in sorted(os.listdir(src_dir)):
        s = os.path.join(src_dir, fn)
        if fn.endswith(".jsonl"):
            with open(s) as fi, open(os.path.join(dst_dir, fn), "w") as fo:
                for line in fi:
                    r = json.loads(line)
                    if r["id"].startswith("gpqa"):
                        r = {k: r[k] for k in ("id", "gold", "k", "score", "preds",
                                               "no_answer", "truncated") if k in r}
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif fn.endswith(".json"):
            shutil.copy(s, os.path.join(dst_dir, fn))


def build(recipe_path: str, artifacts: str, out: str, weights: Optional[str] = None,
          base_eval: Optional[str] = None, model_eval: Optional[str] = None,
          repro: str = "") -> Dict:
    rec = load_recipe(recipe_path)
    res = resolve(rec, artifacts)
    comparison = (reasoning_eval.compare_dirs(base_eval, model_eval)
                  if base_eval and model_eval else None)
    gate = release_gate(res, comparison)
    os.makedirs(out, exist_ok=True)
    if weights:
        for fn in sorted(os.listdir(weights)):
            if (fn.endswith(WEIGHT_FILES) and fn not in SKIP) or fn in ("LICENSE", "NOTICE"):
                shutil.copy(os.path.join(weights, fn), os.path.join(out, fn))
    summaries = {}
    for tag, d in (("base", base_eval), ("model", model_eval)):
        if d:
            _public_eval_copy(d, os.path.join(out, "eval", tag))
            sp = os.path.join(d, "summary.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    summaries[tag] = json.load(f)
    dcfg = {"name": rec.name, "version": rec.version, "base_model": rec.base_model, "hf_repo": rec.hf_repo,
            "components": {k: dict(v["config"], gate=v["numbers"])
                           for k, v in res.included.items()},
            "excluded": {k: v["detail"] for k, v in res.excluded.items()},
            "release_gate": gate}
    with open(os.path.join(out, CONFIG_NAME), "w") as f:
        json.dump(dcfg, f, indent=2)
    if comparison:
        with open(os.path.join(out, "eval", "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(model_card.render(rec, res, gate, comparison, summaries, repro))
    manifest = {}
    for root, _, files in os.walk(out):
        for fn in files:
            if fn != "MANIFEST.json":
                p = os.path.join(root, fn)
                manifest[os.path.relpath(p, out)] = {"sha256": sha256(p),
                                                     "bytes": os.path.getsize(p)}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=2)
    return {"out": out, "gate": gate, "included": sorted(res.included),
            "excluded": {k: v["detail"] for k, v in res.excluded.items()}}


def check_push(folder: str, approved_by: str, public: bool) -> Dict:
    """Refuse to publish unless approved, and refuse public unless ready."""
    if not approved_by or not approved_by.strip():
        raise PermissionError("push needs --approved-by (who approved this upload, and where)")
    with open(os.path.join(folder, CONFIG_NAME)) as f:
        cfg = json.load(f)
    if public and not cfg["release_gate"]["ready"]:
        raise PermissionError("release gate did not pass; only a private upload is allowed: "
                              + "; ".join(cfg["release_gate"]["reasons"]))
    return cfg


def push(folder: str, repo_id: str, approved_by: str, public: bool = False) -> str:
    check_push(folder, approved_by, public)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, private=not public, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id,
                      commit_message=f"DiracLM release upload (approved by: {approved_by})")
    return f"https://huggingface.co/{repo_id}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--recipe", required=True)
    b.add_argument("--artifacts", required=True, help="root the recipe's gate_report paths are under")
    b.add_argument("--out", required=True)
    b.add_argument("--weights")
    b.add_argument("--base-eval")
    b.add_argument("--model-eval")
    b.add_argument("--repro", default="")
    p = sub.add_parser("push")
    p.add_argument("--folder", required=True)
    p.add_argument("--repo", default="")
    p.add_argument("--approved-by", default="")
    p.add_argument("--public", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(a.recipe, a.artifacts, a.out, a.weights, a.base_eval,
                               a.model_eval, a.repro), indent=2))
    else:
        with open(os.path.join(a.folder, CONFIG_NAME)) as f:
            repo = a.repo or json.load(f).get("hf_repo", "")
        if not repo:
            raise SystemExit("--repo is required")
        print(push(a.folder, repo, a.approved_by, a.public))


if __name__ == "__main__":
    main()
