#!/bin/bash
# Ablation: RegisterLM vs SharedAttnOnly vs TinyGPT baseline
set -e
source .venv/bin/activate

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=131072 GRAD_ACCUM_STEPS=1 WARMUP_STEPS=5 TRAIN_LOG_EVERY=50 SEED=1337 MODEL_DIM=256"

echo "=========================================="
echo "1/3: RegisterLM (shared attn x24 + Fourier ops)"
echo "=========================================="
env $COMMON NUM_RECURRENT_STEPS=24 RUN_ID=ablation_registerlm python3 train_register_lm.py

echo ""
echo "=========================================="
echo "2/3: SharedAttnOnly (shared attn x24, NO register ops)"
echo "=========================================="
env $COMMON NUM_RECURRENT_STEPS=24 DISABLE_REGISTER_OPS=1 RUN_ID=ablation_attnonly python3 train_register_lm.py

echo ""
echo "=========================================="
echo "3/3: TinyGPT baseline (1 layer, dim=256, mlp_mult=1)"
echo "=========================================="
env $COMMON NUM_LAYERS=1 MLP_MULT=1 NUM_HEADS=4 NUM_KV_HEADS=2 RUN_ID=ablation_tinygpt python3 train_gpt_mlx.py

echo ""
echo "========== RESULTS =========="
for f in ablation_registerlm ablation_attnonly ablation_tinygpt; do
  echo "--- $f ---"
  grep "^model_params" logs/${f}.txt | head -1
  grep "^step:200" logs/${f}.txt
  grep "^final_int8_zlib_roundtrip " logs/${f}.txt
  echo
done
