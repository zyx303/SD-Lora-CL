# 设置 GPU 设备
export CUDA_VISIBLE_DEVICES=1

# 设置 Hugging Face 模型缓存目录
export HF_HOME=/data/yongxi/.cache/huggingface
export HF_HUB_CACHE=/data/yongxi/.cache/huggingface

# 运行训练
# python3 main.py --config=./exps/sdlora_c100.json  > C100.log 2>&1

python3 main.py --config=./exps/sdlora_inr.json  > InR.log 2>&1 
# python3 main.py --config=./exps/sdlora_ina.json  >> InA.log 2>&1 &
