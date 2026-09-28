# Source this before running Codex's project commands.
# Keep generated files and tool caches inside the repository.
export AMAZON_CODEX_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="$AMAZON_CODEX_WORKSPACE/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export SQLITE_TMPDIR="$TMPDIR"
export POLARS_TEMP_DIR="$TMPDIR"
export JOBLIB_TEMP_FOLDER="$TMPDIR"
export XDG_CACHE_HOME="$AMAZON_CODEX_WORKSPACE/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export npm_config_cache="$XDG_CACHE_HOME/npm"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$AMAZON_CODEX_WORKSPACE/results"
