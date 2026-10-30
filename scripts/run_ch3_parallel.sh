#!/usr/bin/env bash
# Run seed 2 and 3 concurrently; each seed runs four ablations sequentially.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash scripts/run_ch3_parallel.sh GPU_FOR_SEED2 GPU_FOR_SEED3"
  echo "Use two distinct, idle GPUs with verified matching CUDA/Vulkan indices (0,1,2,4)."
  exit 2
fi
GPU_A=$1
GPU_B=$2
for gpu in "$GPU_A" "$GPU_B"; do
  case "$gpu" in 0|1|2|4) ;; *) echo "Unsupported mapping: $gpu. GPUs 3/5 need separate Vulkan mapping validation."; exit 2;; esac
done
[[ "$GPU_A" != "$GPU_B" ]] || { echo 'Choose two different GPUs'; exit 2; }
cd "$(dirname "$0")/.."
source ../.venv-gennbv/bin/activate
export CUDA_HOME="$HOME/.local/opt/cuda-11.7"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/opt/nvidia-550.135/vendor-libs:$CUDA_HOME/lib64"
export VK_ICD_FILENAMES="$HOME/.local/opt/nvidia-550.135/NVIDIA-Linux-x86_64-550.135/nvidia_egl_icd.json"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
unset CUDA_VISIBLE_DEVICES
for file in "$VK_ICD_FILENAMES" ../third_party/hawp/configs/hawpv2.yaml ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth; do
  [[ -f "$file" ]] || { echo "Missing: $file"; exit 2; }
done
for gpu in "$GPU_A" "$GPU_B"; do
  read -r free util <<< "$(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')"
  [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || { echo 'Cannot read GPU state'; exit 2; }
  (( free >= 20000 && util <= 10 )) || { echo "GPU $gpu busy: free=${free}MiB utilization=${util}%. Try another GPU."; exit 2; }
done
# Compile/load once before parallel workers to avoid simultaneous extension builds.
python -c 'from isaacgym import gymapi, gymtorch'
RUN_DIR="_artifacts/ch3_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN_DIR"
printf 'seed2 GPU=%s\nseed3 GPU=%s\n' "$GPU_A" "$GPU_B" > "$RUN_DIR/devices.txt"
echo "Results: $PWD/$RUN_DIR"
pids=()
cleanup() {
  echo 'Interrupted. Stopping this launcher workers.'
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM
for seed in 2 3; do
  gpu=$GPU_A
  [[ "$seed" == 3 ]] && gpu=$GPU_B
  setsid python -u -m chapter3 ablate \
    --stage pilot --seed "$seed" --gpu "$gpu" --graphics "$gpu" \
    --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
    --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
    --output "$RUN_DIR/seed${seed}" > "$RUN_DIR/seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "seed=$seed GPU=$gpu PID=$!"
done
result=0
for i in 0 1; do
  seed=$((i + 2))
  code=0
  wait "${pids[$i]}" || code=$?
  printf 'seed=%s exit_code=%s finished=%s\n' "$seed" "$code" "$(date -Iseconds)" | tee "$RUN_DIR/seed${seed}.status"
  [[ "$code" == 0 ]] || result=1
done
printf '\a'
echo "Both workers exited. Check $RUN_DIR/*.status and seed*/experiment.json"
exit "$result"
