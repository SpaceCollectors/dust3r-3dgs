# Source this before running train.py with gsplat backend:
#   source setup_cuda_env.sh
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH"
export CUDA_HOME="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
export TORCH_CUDA_ARCH_LIST="12.0"
export NVCC_PREPEND_FLAGS="--allow-unsupported-compiler"
echo "CUDA environment configured: CUDA 12.8 + VS 2025 + compute_120"
