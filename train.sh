CUDA_VISIBLE_DEVICES=0 python finetune.py \
--num_epochs 15 \
--lang "hindi" \
--model_name "xlmr" \
--mode "train" \
--base_path "./dataset/"

