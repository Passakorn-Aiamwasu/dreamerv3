#!/usr/bin/env bash
#
# setup_runpod.sh -- provision a RunPod GPU Pod to train DynamicDreamer
# (DreamerV3 + shadow ensemble).
#
# Usage (first boot and every subsequent Pod restart):
#   cd /workspace/DynamicDreamer   # wherever this repo is cloned
#   bash setup_runpod.sh
#
# Flags:
#   --force   Rebuild .venv from scratch even if it already has the right
#             jax version installed. Use this after changing
#             requirements-runpod.txt.
#
# Safe to re-run: on a Pod whose /workspace is a persistent network
# volume, .venv survives a Pod restart, and a second run skips venv
# creation/reinstall entirely once it confirms jax==0.4.33 is already
# there -- it just re-activates and re-verifies. GitHub SSH access is
# NOT handled here; see docs/GITHUB_SSH_SETUP.md for that bootstrap step
# (kept separate on purpose -- it touches private key material, which
# does not belong in an automated script).

set -uo pipefail
# Deliberately no `-e`: the verification steps (9-11 below) run every
# check and report a full pass/fail summary before the script decides
# whether to exit non-zero, rather than dying on the first failure and
# hiding what else would also have failed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements-runpod.txt"
JAX_VERSION="0.4.33"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help)
      echo "Usage: bash setup_runpod.sh [--force]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (see --help)" >&2
      exit 1
      ;;
  esac
done

log() { echo "[setup_runpod] $*"; }
die() { echo "[setup_runpod] ERROR: $*" >&2; exit 1; }

# ============================================================
# 1. GPU / CUDA check
# ============================================================
log "Checking for a GPU..."
command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found -- this Pod has no NVIDIA GPU/driver."
nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi found but failed to run -- driver/GPU not accessible."
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
log "GPU: $GPU_NAME (driver $DRIVER_VERSION)"

# ============================================================
# 2/3. Create + activate .venv
# ============================================================
NEED_INSTALL=1
if [ -d "$VENV_DIR" ] && [ "$FORCE" -eq 0 ]; then
  INSTALLED_JAX="$("$VENV_DIR/bin/python" -c 'import jax; print(jax.__version__)' 2>/dev/null || true)"
  if [ "$INSTALLED_JAX" = "$JAX_VERSION" ]; then
    log ".venv already has jax==$JAX_VERSION, skipping install (use --force to rebuild)."
    NEED_INSTALL=0
  else
    log ".venv exists but jax is '${INSTALLED_JAX:-<not installed>}' (want $JAX_VERSION) -- reinstalling."
  fi
fi

if [ ! -d "$VENV_DIR" ] || [ "$FORCE" -eq 1 ]; then
  log "Creating .venv..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR" || die "python3 -m venv failed."
  NEED_INSTALL=1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Activated $VENV_DIR (python: $(python -V 2>&1))"

# ============================================================
# 4/5. Install JAX 0.4.33 + CUDA, MuJoCo, dm_control, everything else
# ============================================================
if [ "$NEED_INSTALL" -eq 1 ]; then
  log "Upgrading pip..."
  pip install --quiet --upgrade pip || die "pip upgrade failed."
  log "Installing from $(basename "$REQUIREMENTS")..."
  pip install --quiet -r "$REQUIREMENTS" || die "pip install -r $REQUIREMENTS failed."
else
  log "Skipping pip install (already satisfied)."
fi

# ============================================================
# 6. EGL libraries (system packages, for headless MuJoCo rendering)
# ============================================================
log "Checking EGL system libraries..."
EGL_PACKAGES=(libegl1 libgl1 libopengl0 libosmesa6)
if ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
  log "libEGL already present."
elif [ "$(id -u)" -eq 0 ]; then
  log "Installing ${EGL_PACKAGES[*]} via apt-get..."
  if ! apt-get update -qq; then
    die "apt-get update failed."
  fi
  if ! apt-get install -y -qq "${EGL_PACKAGES[@]}"; then
    die "apt-get install of EGL libraries failed."
  fi
elif command -v sudo >/dev/null 2>&1; then
  log "Installing ${EGL_PACKAGES[*]} via sudo apt-get..."
  if ! sudo apt-get update -qq; then
    die "apt-get update failed."
  fi
  if ! sudo apt-get install -y -qq "${EGL_PACKAGES[@]}"; then
    die "apt-get install of EGL libraries failed."
  fi
else
  die "libEGL missing and neither root nor sudo is available to install ${EGL_PACKAGES[*]}."
fi

# ============================================================
# 7/8. Environment variables -- exported for this shell, persisted into
# .venv/bin/activate (so every future `source .venv/bin/activate` sets
# them too), and written to .env for anything that doesn't go through
# the venv's own activate script.
# ============================================================
export MUJOCO_GL=egl
export JAX_PLATFORMS=cuda

cat > "$SCRIPT_DIR/.env" <<'EOF'
export MUJOCO_GL=egl
export JAX_PLATFORMS=cuda
EOF

if ! grep -q "DynamicDreamer RunPod env" "$VENV_DIR/bin/activate" 2>/dev/null; then
  {
    echo ""
    echo "# --- DynamicDreamer RunPod env (added by setup_runpod.sh) ---"
    echo "export MUJOCO_GL=egl"
    echo "export JAX_PLATFORMS=cuda"
  } >> "$VENV_DIR/bin/activate"
fi
log "Set MUJOCO_GL=egl, JAX_PLATFORMS=cuda (persisted in .venv/bin/activate and .env)"

# ============================================================
# 9/10/11. Verification
# ============================================================
declare -A RESULTS
PASS="OK"
FAIL="FAIL"

check_jax_gpu() {
  python - <<'PY'
import sys
try:
  import jax
  devices = jax.devices()
  gpu = [d for d in devices if d.platform in ('gpu', 'cuda')]
  print("JAX_DEVICES=" + repr(devices))
  sys.exit(0 if gpu else 1)
except Exception as e:
  print("JAX_ERROR=" + str(e))
  sys.exit(1)
PY
}
if JAX_DEVICES_LINE="$(check_jax_gpu)"; then
  RESULTS[jax]="$PASS"
else
  RESULTS[jax]="$FAIL"
fi
JAX_DEVICES_STR="$(echo "$JAX_DEVICES_LINE" | sed -n 's/^JAX_DEVICES=//p')"

if python -c "import mujoco; mujoco.MjModel.from_xml_string('<mujoco/>')" >/dev/null 2>&1; then
  RESULTS[mujoco]="$PASS"
else
  RESULTS[mujoco]="$FAIL"
fi

if python -c "import dm_control.suite" >/dev/null 2>&1; then
  RESULTS[dmc]="$PASS"
else
  RESULTS[dmc]="$FAIL"
fi

if MUJOCO_GL=egl python -c "
import mujoco
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type=\"sphere\" size=\"1\"/></worldbody></mujoco>')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 64, 64)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
renderer.render()
" >/dev/null 2>&1; then
  RESULTS[egl]="$PASS"
else
  RESULTS[egl]="$FAIL"
fi

log "Running smoke test (test_shadow_agent_loss.py)..."
if python test_shadow_agent_loss.py > /tmp/setup_runpod_smoke.log 2>&1; then
  RESULTS[smoke]="$PASS"
else
  RESULTS[smoke]="$FAIL"
  log "Smoke test failed -- see /tmp/setup_runpod_smoke.log"
fi

PYTHON_VERSION="$(python -V 2>&1 | awk '{print $2}')"
JAX_VERSION_INSTALLED="$(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'unknown')"

echo ""
echo "========================================"
echo " DynamicDreamer RunPod Environment"
echo "========================================"
echo ""
echo "Python:"
echo "    $PYTHON_VERSION"
echo ""
echo "GPU:"
echo "    $GPU_NAME"
echo ""
echo "JAX:"
echo "    $JAX_VERSION_INSTALLED"
echo ""
echo "JAX devices:"
echo "    ${JAX_DEVICES_STR:-<unavailable>}"
echo ""
echo "MuJoCo:"
echo "    ${RESULTS[mujoco]}"
echo ""
echo "dm-control:"
echo "    ${RESULTS[dmc]}"
echo ""
echo "EGL:"
echo "    ${RESULTS[egl]}"
echo ""
echo "Smoke test (test_shadow_agent_loss.py):"
echo "    ${RESULTS[smoke]}"
echo ""
echo "Environment:"
echo "    MUJOCO_GL=$MUJOCO_GL"
echo "    JAX_PLATFORMS=$JAX_PLATFORMS"
echo ""
echo "========================================"

OVERALL="$PASS"
for key in jax mujoco dmc egl smoke; do
  [ "${RESULTS[$key]}" = "$PASS" ] || OVERALL="$FAIL"
done

if [ "$OVERALL" = "$PASS" ]; then
  echo " Setup complete"
  echo "========================================"
  exit 0
else
  echo " Setup FAILED -- see FAIL entries above"
  echo "========================================"
  exit 1
fi
