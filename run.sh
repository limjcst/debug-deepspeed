deepspeed --num_gpus=2 \
    train.py \
    --num_stages 2 \
    --steps 10 \
    --deepspeed \
    --deepspeed_config=ds_config.json
