#!/bin/bash
#PBS -k doe -j oe
source /work/zb023/research/ftwp-2026/scripts/hpc_common.sh
TQDM_DISABLE=1 python3 run_navigator_raw_cmd.py -nav -rawcmd