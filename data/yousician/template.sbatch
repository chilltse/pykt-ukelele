#!/bin/bash
#SBATCH -J aktint_1c_predict
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# 1) 切到你要跑的目录
cd ~/my_code/pykt-ukelele/examples

# 2) 激活 conda 环境
source ~/.bashrc
source /home/remote/u8016457/miniconda3/etc/profile.d/conda.sh
conda activate pykt

echo "=== Job info ==="
hostname
pwd
which python
python -V
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 3) 执行命令