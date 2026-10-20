#!/usr/bin/env bash
# Stage 1 of extraction-distill: check the GPU inside WSL Ubuntu and install a PyTorch
# venv (in the WSL filesystem, not /mnt/c, which is slow). Safe to re-run.
set -uo pipefail
LOG=/mnt/c/code/extraction-distill/logs/setup_wsl.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

echo "== distro"; . /etc/os-release; echo "$PRETTY_NAME"; uname -r
echo "cpus: $(nproc)"; free -g | head -2; df -h "$HOME" | tail -1

echo "== gpu in WSL"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || echo "NO GPU IN WSL"

echo "== uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv --version || { echo "uv install failed"; exit 1; }

echo "== torch venv"
V="$HOME/venvs/distill-train"
[ -d "$V" ] || uv venv --python 3.12 "$V"
. "$V/bin/activate"
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import time, torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
    print("arch list", torch.cuda.get_arch_list())
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = a @ b; torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"bf16 matmul: {20 * 2 * 8192**3 / dt / 1e12:.1f} TFLOPS")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free/total GB: {free/1e9:.1f} / {total/1e9:.1f}")
PY
echo "DONE"
