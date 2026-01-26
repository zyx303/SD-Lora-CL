export CUDA_VISIBLE_DEVICES=6,7
python3 main.py --config=./exps/sdlora_c100.json  > C100.log 2>&1

# python3 main.py --config=./exps/sdlora_inr.json  >> InR.log 2>&1 &
# python3 main.py --config=./exps/sdlora_ina.json  >> InA.log 2>&1 &
