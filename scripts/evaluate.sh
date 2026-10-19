#!/usr/bin/env bash

set -uo pipefail
shopt -s nullglob

usage() {
  cat <<'EOF'
Evaluate every completed run in one experiment directory or in all experiments.

Usage:
  scripts/evaluate.sh --exp-dir EXP_DIR --data DATA [DATA ...] [EVALUATE_OPTIONS]
  scripts/evaluate.sh --all [--outputs-dir OUTPUTS_DIR] --data DATA [DATA ...] [EVALUATE_OPTIONS]

Examples:
  scripts/evaluate.sh \
    --exp-dir outputs/captcha100k-resnet18-aug \
    --data data/test/test4

  scripts/evaluate.sh \
    --all \
    --data data/test/test0 data/test/test1 data/test/test2 \
    --device cuda

All arguments after the script options are forwarded to scripts/evaluate.py.
Use --outputs-dir to change the directory scanned by --all (default: outputs).
EOF
}

experiment_dir=""
outputs_dir="outputs"
evaluate_all=false
evaluate_args=()

while (($#)); do
  case "$1" in
    --exp-dir)
      if (($# < 2)); then
        echo "error: --exp-dir requires a directory" >&2
        usage >&2
        exit 2
      fi
      experiment_dir=$2
      shift 2
      ;;
    --all)
      evaluate_all=true
      shift
      ;;
    --outputs-dir)
      if (($# < 2)); then
        echo "error: --outputs-dir requires a directory" >&2
        usage >&2
        exit 2
      fi
      outputs_dir=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      evaluate_args+=("$@")
      break
      ;;
    *)
      # The remaining options belong to evaluate.py, beginning with --data.
      evaluate_args+=("$@")
      break
      ;;
  esac
done

if [[ $evaluate_all == true && -n $experiment_dir ]]; then
  echo "error: choose either --exp-dir or --all, not both" >&2
  exit 2
fi
if [[ $evaluate_all == false && -z $experiment_dir ]]; then
  echo "error: either --exp-dir or --all is required" >&2
  usage >&2
  exit 2
fi

has_data=false
for argument in "${evaluate_args[@]}"; do
  if [[ $argument == --data ]]; then
    has_data=true
    break
  fi
done
if [[ $has_data == false ]]; then
  echo "error: pass at least one evaluation dataset with --data" >&2
  usage >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

run_dirs=()
if [[ $evaluate_all == true ]]; then
  if [[ ! -d $outputs_dir ]]; then
    echo "error: outputs directory does not exist: $outputs_dir" >&2
    exit 2
  fi
  for config_path in "$outputs_dir"/*/*/config.json; do
    run_dirs+=("${config_path%/config.json}")
  done
else
  if [[ ! -d $experiment_dir ]]; then
    echo "error: experiment directory does not exist: $experiment_dir" >&2
    exit 2
  fi
  for config_path in "$experiment_dir"/*/config.json; do
    run_dirs+=("${config_path%/config.json}")
  done
fi

if ((${#run_dirs[@]} == 0)); then
  echo "error: no run directories containing config.json were found" >&2
  exit 1
fi

python_command=${PYTHON:-python}
attempted=0
succeeded=0
failed=0
skipped=0

for run_dir in "${run_dirs[@]}"; do
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f $checkpoint ]]; then
    echo "skip: no best checkpoint in $run_dir" >&2
    ((skipped += 1))
    continue
  fi

  ((attempted += 1))
  echo
  echo "==> Evaluating $run_dir"
  if "$python_command" scripts/evaluate.py --checkpoint "$checkpoint" "${evaluate_args[@]}"; then
    ((succeeded += 1))
  else
    echo "failed: $run_dir" >&2
    ((failed += 1))
  fi
done

echo
echo "Evaluation summary: $succeeded succeeded, $failed failed, $skipped skipped."

if ((attempted == 0 || failed > 0)); then
  exit 1
fi
