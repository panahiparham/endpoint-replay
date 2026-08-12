#!/usr/bin/env bash
# Install the ale-py XLA build for Atari support on macOS / Linux (CPU or CUDA).
#
#   ./scripts/install_ale_wheel.sh            # macOS-CPU / Linux-CPU
#   ./scripts/install_ale_wheel.sh --cuda     # Linux-CUDA (also installs jax[cuda12])
#
# Why this exists: PyPI's ale-py 0.12.0 predates ALE PR #707 ("Enable XLA on MacOS
# and fix XLA on GPUs"). On macOS it ships no vector-XLA FFI at all
# (AtariVectorEnv(...).xla() -> AttributeError on VectorXLAReset); on Linux it has
# the CPU FFI but no CUDA targets, so a GPU run dies at execution. So we `uv sync`
# the (optional) atari extra, then FORCE-INSTALL the PR #707 wheel over whatever
# PyPI build uv put in place. Run this AFTER any `uv sync --extra atari` -- a bare
# sync silently reinstalls the broken PyPI build (both report version 0.12.0).
#
# ARTIFACT EXPIRY: the PR #707 CI artifacts were created 2026-06-28 and EXPIRE
# 2026-09-26. After that this script cannot download them; a machine with an empty
# $WHEEL_CACHE (default ~/.cache/ale-py-pr707-wheels) cannot bootstrap, so preserve
# that cache. Longer term, replace this with a PyPI ale-py release that includes #707.
#
# Requires a GitHub token for the first fetch: an authenticated `gh` CLI, or
# GH_TOKEN=<PAT>. Subsequent runs reuse $WHEEL_CACHE (no token/network needed).
# GSP_VENV=/path/to/.venv targets an external environment instead of $PROJECT/.venv.
# The cluster scripts set it: there the venv is shared by every run dir with the same
# uv.lock, so it is deps-only and the project must NOT be installed into it.
set -euo pipefail

CUDA=0
[ "${1:-}" = "--cuda" ] && CUDA=1

REPO="Farama-Foundation/Arcade-Learning-Environment"
RUN_ID="28333756257"   # CI run for PR #707
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_CACHE="${WHEEL_CACHE:-$HOME/.cache/ale-py-pr707-wheels}"

case "$(uname -s)" in
  Darwin) ARTIFACT="wheels-macOS-arm64";   GLOB="ale_py-*-cp313-cp313-macosx_*_arm64.whl" ;;
  Linux)  ARTIFACT="wheels-Linux-x86_64";  GLOB="ale_py-*-cp313-cp313-manylinux*x86_64.whl" ;;
  *) echo "error: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

command -v uv >/dev/null || { echo "error: uv not found" >&2; exit 1; }
mkdir -p "$WHEEL_CACHE"
have() { find "$1" -maxdepth 1 -name "$GLOB" 2>/dev/null | head -n1; }

wheel="$(have "$WHEEL_CACHE")"
if [ -z "$wheel" ]; then
  echo ">> $ARTIFACT: not cached, fetching from ALE PR #707 CI ..."
  tmp="$(mktemp -d)"
  if command -v gh >/dev/null; then
    gh run download "$RUN_ID" --repo "$REPO" --name "$ARTIFACT" --dir "$tmp"
  elif [ -n "${GH_TOKEN:-}" ]; then
    art_id="$(curl -fsSL -H "Authorization: Bearer $GH_TOKEN" \
      "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/artifacts?per_page=100" \
      | uv run --no-project python -c "import json,sys; a=[x for x in json.load(sys.stdin)['artifacts'] if x['name']=='$ARTIFACT' and not x['expired']]; sys.exit('no live artifact $ARTIFACT (expired?)') if not a else print(a[0]['id'])")"
    curl -fsSL -H "Authorization: Bearer $GH_TOKEN" -o "$tmp/w.zip" \
      "https://api.github.com/repos/$REPO/actions/artifacts/$art_id/zip"
    ( cd "$tmp" && unzip -q -o w.zip )
  else
    echo "error: authenticate gh or set GH_TOKEN, or place a wheel in $WHEEL_CACHE" >&2
    exit 1
  fi
  found="$(have "$tmp")"
  [ -n "$found" ] || { echo "error: no wheel matching '$GLOB' in $ARTIFACT" >&2; exit 1; }
  cp -f "$found" "$WHEEL_CACHE/"
  wheel="$WHEEL_CACHE/$(basename "$found")"
  rm -rf "$tmp"
else
  echo ">> reusing $(basename "$wheel") from $WHEEL_CACHE"
fi

EXTRAS=(--extra atari)
[ "$CUDA" = 1 ] && EXTRAS+=(--extra cuda)

# Targeting a shared, deps-only venv (the cluster case) vs. this checkout's own.
SYNC_FLAGS=()
PIP_TARGET=()
if [ -n "${GSP_VENV:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="$GSP_VENV"
  SYNC_FLAGS=(--frozen --no-install-project)
  PIP_TARGET=(--python "$GSP_VENV/bin/python")
  echo ">> targeting the shared env at $GSP_VENV"
fi

echo ">> uv sync ${EXTRAS[*]} ${SYNC_FLAGS[*]-} ..."
( cd "$PROJECT" && uv sync "${EXTRAS[@]}" ${SYNC_FLAGS[@]+"${SYNC_FLAGS[@]}"} )

echo ">> force-installing the PR #707 wheel over the PyPI build ..."
( cd "$PROJECT" && uv pip install ${PIP_TARGET[@]+"${PIP_TARGET[@]}"} --reinstall --no-deps "$wheel" )

echo ">> verifying the XLA FFI is compiled in and pong resets ..."
( cd "$PROJECT" && ALE_VERIFY_CUDA="$CUDA" uv run --no-sync python - <<'PY'
import importlib.util, os, sys, ale_py, ale_py._ale_py as c
missing = [n for n in ("VectorXLAReset", "VectorXLAStep") if not hasattr(c, n)]
if missing:
    sys.exit(f"error: ale-py {ale_py.__version__} missing {missing} -> not a PR #707 build")

# With --cuda the wheel must carry the CUDA FFI targets AND jax must have its CUDA
# plugin. ale-py's GPU FFI lives in the wheel independent of jax's plugin, and the
# reset check below runs on CPU, so a CPU-only wheel or a pruned jax-cuda12-plugin
# (e.g. a later bare `uv sync` dropping --extra cuda) would sail through green while
# GPU jobs silently ran on CPU. Fail fast on either so cluster runs use real CUDA XLA.
if os.environ.get("ALE_VERIFY_CUDA") == "1":
    gpu_missing = [n for n in ("VectorXLAResetGPU", "VectorXLAStepGPU") if not hasattr(c, n)]
    if gpu_missing:
        sys.exit(f"error: ale-py {ale_py.__version__} missing {gpu_missing} -> a CPU-only wheel, "
                 "not the CUDA PR #707 build; --cuda Atari would run on CPU")
    if importlib.util.find_spec("jax_plugins.xla_cuda12") is None:
        sys.exit("error: jax-cuda12-plugin not installed -> GPU jax is missing (was --extra cuda "
                 "pruned by a bare `uv sync`?); --cuda Atari would fall back to CPU")

import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
from ale_py import AtariVectorEnv
h, reset_fn, _ = AtariVectorEnv("pong", num_envs=1).xla()
reset_fn(h, jnp.array([0], jnp.int32))
tag = "CPU+CUDA" if os.environ.get("ALE_VERIFY_CUDA") == "1" else "CPU"
print(f"ok: ale-py {ale_py.__version__} (PR #707), {tag} Vector XLA FFI present, pong resets")
PY
)
