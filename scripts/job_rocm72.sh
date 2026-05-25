# 1. 老师写的修复代码放在这里（精准拦截后续传入的错误值）
case ${CUDA_VISIBLE_DEVICES} in
    0-*)
        echo "CUDA_VISIBLE_DEVICES has broken value: ${CUDA_VISIBLE_DEVICES}, fixing to 0"
        CUDA_VISIBLE_DEVICES=0
        ;;
    1-*)
        echo "CUDA_VISIBLE_DEVICES has broken value: ${CUDA_VISIBLE_DEVICES}, fixing to 1"
        CUDA_VISIBLE_DEVICES=1
        ;;
esac
export CUDA_VISIBLE_DEVICES

HF_HOME=/work/${LOGNAME}/.cache/huggingface
TORCH_HOME=/work/${LOGNAME}/.cache/torch
TRITON_CACHE_DIR=/work/${LOGNAME}/.cache/triton
UV_CACHE_DIR=/work/${LOGNAME}/.cache/uv
export HF_HOME TORCH_HOME TRITON_CACHE_DIR UV_CACHE_DIR

cd /work/zb023/micromamba
eval "$(./bin/micromamba shell activate -p ./envs/torch_rocm72)"
cd /work/zb023/research/ftwp-2026
python3 run.py