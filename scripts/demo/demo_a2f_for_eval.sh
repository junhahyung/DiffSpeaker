export CUDA_VISIBLE_DEVICES=1

# # use hubert backbone
# python demo_vocaset.py \
#     --cfg configs/diffusion/vocaset/diffspeaker_hubert_vocaset.yaml \
#     --cfg_assets configs/assets/vocaset.yaml \
#     --template datasets/vocaset/templates.pkl \
#     --example demo/wavs/speech_long.wav \
#     --ply datasets/vocaset/templates/FLAME_sample.ply \
#     --checkpoint checkpoints/vocaset/diffspeaker_hubert_vocaset.ckpt \
#     --id FaceTalk_170809_00138_TA

# use wav2vec2 backbone
#--example demo/wavs/speech_british.wav \
python demo_a2f_for_eval.py \
    --cfg configs/diffusion/a2f/diffspeaker_hubert_a2f.yaml \
    --cfg_assets configs/assets/a2f.yaml \
    --ply data/A2F/face_template.obj \
    --checkpoint experiments/a2f/diffusion_bias/diffspeaker_hubert_a2f/checkpoints/epoch=499.ckpt \

