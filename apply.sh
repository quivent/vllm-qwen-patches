#!/usr/bin/env bash
# Apply vLLM 0.19.0 patches for Qwen 3.5-27B speculative decoding.
#
# Usage:
#   ./apply.sh [patch_name]    Apply a specific patch
#   ./apply.sh all             Apply all patches
#   ./apply.sh list            Show available patches
#   ./apply.sh check           Verify vLLM version and paths
#   ./apply.sh revert          Revert all patches (from .bak files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find vLLM installation
find_vllm() {
    python3 -c "import vllm; import os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null
}

VLLM_ROOT="${VLLM_ROOT:-$(find_vllm)}"
if [ -z "$VLLM_ROOT" ]; then
    echo "error: cannot find vLLM installation. Set VLLM_ROOT or install vllm."
    exit 1
fi

VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "vLLM $VLLM_VERSION at $VLLM_ROOT"

# Patch definitions: name → description → file → type (patch|copy)
declare -A PATCH_DESC=(
    [eagle]="5 fixes for propose_tree() on MTP + M-RoPE multimodal models"
    [qwen3_next]="tensor shape fix for compiled forward in modal_mtp draft mode"
    [speculative]="guard hf_config_override so standalone draft models aren't forced into MTP"
    [modal_mtp]="DeltaNet self-speculative proposer (new file, 3 bug fixes baked in)"
)

apply_patch() {
    local name="$1"
    case "$name" in
        eagle)
            local target="$VLLM_ROOT/vllm/v1/spec_decode/eagle.py"
            cp "$target" "$target.bak" 2>/dev/null || true
            cd "$VLLM_ROOT" && patch -p1 --forward < "$SCRIPT_DIR/eagle.patch"
            echo "  applied: eagle.py (5 fixes)"
            ;;
        qwen3_next)
            local target="$VLLM_ROOT/vllm/model_executor/models/qwen3_next.py"
            cp "$target" "$target.bak" 2>/dev/null || true
            cd "$VLLM_ROOT" && patch -p1 --forward < "$SCRIPT_DIR/qwen3_next.patch"
            echo "  applied: qwen3_next.py"
            ;;
        speculative)
            local target="$VLLM_ROOT/vllm/config/speculative.py"
            cp "$target" "$target.bak" 2>/dev/null || true
            cd "$VLLM_ROOT" && patch -p1 --forward < "$SCRIPT_DIR/speculative-draft-override.patch"
            echo "  applied: speculative.py"
            echo "  WARNING: this patch changes MTP config detection. Test your config before deploying."
            ;;
        modal_mtp)
            local target="$VLLM_ROOT/vllm/v1/spec_decode/modal_mtp.py"
            cp "$target" "$target.bak" 2>/dev/null || true
            cp "$SCRIPT_DIR/modal_mtp.py" "$target"
            echo "  applied: modal_mtp.py (copied)"
            echo "  NOTE: modal_mtp has a known DeltaNet state corruption issue. See README."
            ;;
        *)
            echo "error: unknown patch '$name'"
            echo "available: eagle, qwen3_next, speculative, modal_mtp"
            return 1
            ;;
    esac
}

revert_patch() {
    local name="$1"
    case "$name" in
        eagle)       local target="$VLLM_ROOT/vllm/v1/spec_decode/eagle.py" ;;
        qwen3_next)  local target="$VLLM_ROOT/vllm/model_executor/models/qwen3_next.py" ;;
        speculative) local target="$VLLM_ROOT/vllm/config/speculative.py" ;;
        modal_mtp)   local target="$VLLM_ROOT/vllm/v1/spec_decode/modal_mtp.py" ;;
        *) echo "unknown: $name"; return 1 ;;
    esac
    if [ -f "$target.bak" ]; then
        cp "$target.bak" "$target"
        echo "  reverted: $name"
    else
        echo "  skip: no .bak for $name"
    fi
}

case "${1:-list}" in
    list)
        echo ""
        echo "Available patches:"
        for name in eagle qwen3_next speculative modal_mtp; do
            echo "  $name — ${PATCH_DESC[$name]}"
        done
        echo ""
        echo "Usage: ./apply.sh <patch_name|all|check|revert>"
        ;;
    check)
        echo "version: $VLLM_VERSION"
        echo "root:    $VLLM_ROOT"
        for name in eagle qwen3_next speculative modal_mtp; do
            case "$name" in
                eagle)       f="$VLLM_ROOT/vllm/v1/spec_decode/eagle.py" ;;
                qwen3_next)  f="$VLLM_ROOT/vllm/model_executor/models/qwen3_next.py" ;;
                speculative) f="$VLLM_ROOT/vllm/config/speculative.py" ;;
                modal_mtp)   f="$VLLM_ROOT/vllm/v1/spec_decode/modal_mtp.py" ;;
            esac
            if [ -f "$f.bak" ]; then
                echo "  $name: PATCHED (backup exists)"
            elif [ -f "$f" ]; then
                echo "  $name: stock"
            else
                echo "  $name: FILE MISSING"
            fi
        done
        ;;
    all)
        echo "Applying all patches..."
        for name in eagle qwen3_next; do
            apply_patch "$name"
        done
        echo ""
        echo "Safe patches applied (eagle, qwen3_next)."
        echo "Optional patches (run separately if needed):"
        echo "  ./apply.sh speculative  — for standalone draft model support"
        echo "  ./apply.sh modal_mtp    — for DeltaNet self-speculative (experimental)"
        ;;
    revert)
        echo "Reverting all patches..."
        for name in eagle qwen3_next speculative modal_mtp; do
            revert_patch "$name"
        done
        ;;
    *)
        apply_patch "$1"
        ;;
esac
