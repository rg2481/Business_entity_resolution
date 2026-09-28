# Keep every tool's temp files and caches inside this repo.
# Usage (at the start of each shell command that runs Python or other tools):
#   source matcher/env.sh      (from the repository / package root)
export CLAUDE_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TMPDIR="$CLAUDE_WS/tmp" TMP="$CLAUDE_WS/tmp" TEMP="$CLAUDE_WS/tmp"
export POLARS_TEMP_DIR="$CLAUDE_WS/tmp"
export JOBLIB_TEMP_FOLDER="$CLAUDE_WS/tmp" SQLITE_TMPDIR="$CLAUDE_WS/tmp"
export XDG_CACHE_HOME="$CLAUDE_WS/cache"
export PIP_CACHE_DIR="$CLAUDE_WS/cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$CLAUDE_WS/cache/huggingface"
export TORCH_HOME="$CLAUDE_WS/cache/torch"
export TRITON_CACHE_DIR="$CLAUDE_WS/cache/triton"
export CUDA_CACHE_PATH="$CLAUDE_WS/cache/nv"
export MPLCONFIGDIR="$CLAUDE_WS/cache/matplotlib"
export NUMBA_CACHE_DIR="$CLAUDE_WS/cache/numba"
export PYTHONPYCACHEPREFIX="$CLAUDE_WS/cache/pycache"
export npm_config_cache="$CLAUDE_WS/cache/npm"

# Extra Python packages go here (pip install --no-cache-dir --target "$CLAUDE_WS/pylib" <pkg>),
# never into ~/.local. Existing packages in ~/.local stay importable.
export PYTHONPATH="$CLAUDE_WS/pylib${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CLAUDE_WS/pylib"
