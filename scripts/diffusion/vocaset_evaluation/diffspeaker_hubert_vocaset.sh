export CUDA_VISIBLE_DEVICES=0
python eval_vocaset.py \
    --cfg configs/diffusion/biwi/diffspeaker_hubert_vocaset.yaml \
    --cfg_assets configs/assets/vocaset.yaml \
