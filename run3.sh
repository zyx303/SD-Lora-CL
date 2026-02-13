# 设置 GPU 设备
export CUDA_VISIBLE_DEVICES=3


# 强制使用离线模式，避免 SSL 连接错误
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 运行训练
# python3 main.py --config=./exps/sdlora_c100.json  > C100.log 2>&1

python3 main.py --config=./exps/sdlora_ca_inr.json  >> InR_CA.log 2>&1 
# python3 main.py --config=./exps/sdlora_ina.json  >> InA.log 2>&1 &
