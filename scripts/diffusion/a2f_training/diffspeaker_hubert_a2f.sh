export CUDA_VISIBLE_DEVICES=0,1
python -m train \
    --cfg configs/diffusion/a2f/diffspeaker_hubert_a2f.yaml \
    --cfg_assets configs/assets/a2f.yaml \
    --batch_size 8 \
    --nodebug

