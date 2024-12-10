import numpy as np
import os
import sys
sys.path.append('/workspace/repos/facediffuser')
from data_utils.A2F_data_utils import read_shot_list, normalize_data
import shutil
import pickle

dataset_extended_claire = {
    'actor_mesh': '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/skin',
    'shots_root': '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/audios',
    'shots': [ # (shot_path, shot_fps, shot_start, shot_end)
        ("{0}/{0}.wav".format("cp1_neutral"), 30.0,  25, 330, 1),  #349
        ("{0}/{0}.wav".format("cp2_neutral"), 30.0,  20, 320, 1),  #344
        ("{0}/{0}.wav".format("cp3_neutral"), 30.0,  15, 335, 1),  #338
        ("{0}/{0}.wav".format("cp4_neutral"), 30.0,  25, 335, 1),  #356
        ("{0}/{0}.wav".format("cp5_neutral"), 30.0,  30, 300, 1),  #320
        ("{0}/{0}.wav".format("cp6_neutral"), 30.0,  30, 250, 4),  #284
        ("{0}/{0}.wav".format("cp7_neutral"), 30.0,  20, 245, 0),  #251
        ("{0}/{0}.wav".format("cp8_neutral"), 30.0,  25, 340, 1),  #350
        ("{0}/{0}.wav".format("cp9_neutral"), 30.0,  35, 385, 1),  #397
        ("{0}/{0}.wav".format("cp10_neutral"), 30.0,  30, 315, 0),  #319
        ("{0}/{0}.wav".format("cp11_neutral"), 30.0,  20, 319, 1),  #319
        ("{0}/{0}.wav".format("cp12_neutral"), 30.0,  25, 333, 1),  #333
        ("{0}/{0}.wav".format("cp13_neutral"), 30.0,  20, 290, 1),  #296
        ("{0}/{0}.wav".format("cp14_neutral"), 30.0,  15, 280, 1),  #303
        ("{0}/{0}.wav".format("cp15_neutral"), 30.0,  25, 360, 0),  #403
        ("{0}/{0}.wav".format("cp16_neutral"), 30.0,  35, 390, 1),  #390
        ("{0}/{0}.wav".format("cp17_neutral"), 30.0,  40, 330, 0),  #373
        ("{0}/{0}.wav".format("cp18_neutral"), 30.0,  25, 270, 0),  #280
        ("{0}/{0}.wav".format("cp19_neutral"), 30.0,  60, 255, 0),  #281
        ("{0}/{0}.wav".format("cp20_neutral"), 30.0,  30, 280, 0),  #289
        ("{0}/{0}.wav".format("cp21_neutral"), 30.0,  30, 505, 0),  #551
        ("{0}/{0}.wav".format("cp22_amazement"), 30.0,  25, 270, 0),  #281
        ("{0}/{0}.wav".format("cp23_amazement"), 30.0,  20, 480, 1),  #527
        ("{0}/{0}.wav".format("cp24_joy"), 30.0,  30, 273, 1),  #273
        ("{0}/{0}.wav".format("cp25_joy"), 30.0,  25, 455, 1),  #495
        ("{0}/{0}.wav".format("cp26_cheekiness"), 30.0,  20, 280, 1),  #304
        ("{0}/{0}.wav".format("cp27_cheekiness"), 30.0,  40, 540, 1),  #575
        ("{0}/{0}.wav".format("cp28_sadness"), 30.0,  40, 270, 1),  #273
        ("{0}/{0}.wav".format("cp29_sadness"), 30.0,  45, 650, 1),  #711
        ("{0}/{0}.wav".format("cp30_disgust"), 30.0,  20, 290, 1),  #317
        ("{0}/{0}.wav".format("cp31_disgust"), 30.0,  25, 525, 1),  #682
        ("{0}/{0}.wav".format("cp32_anger"), 30.0,  30, 235, 0),  #239
        ("{0}/{0}.wav".format("cp33_anger"), 30.0,  25, 400, 1),  #504
        ("{0}/{0}.wav".format("cp34_fear"), 30.0,  20, 280, 1),  #287
        ("{0}/{0}.wav".format("cp35_fear"), 30.0,  30, 540, 1),  #596
        ("{0}/{0}.wav".format("cp36_grief"), 30.0,  45, 370, 1),  #371
        ("{0}/{0}.wav".format("cp37_grief"), 30.0,  35, 630, 1),  #661
        ("{0}/{0}.wav".format("cp38_pain"), 30.0,  40, 285, 0),  #297
        ("{0}/{0}.wav".format("cp39_pain"), 30.0,  30, 490, 0),  #549
        ("{0}/{0}.wav".format("cp40_outofbreath"), 30.0,  45, 270, 0),  #293
        ("{0}/{0}.wav".format("cp41_outofbreath"), 30.0,  30, 470, 1),  #504
        ("{0}/{0}.wav".format("cp42_fastneutral"), 30.0,  15, 132, 0),  #132
        ("{0}/{0}.wav".format("cp43_fastneutral"), 30.0,  20, 250, 1),  #271
        ("{0}/{0}.wav".format("cp44_slowneutral"), 30.0,  30, 371, 1),  #371
        ("{0}/{0}.wav".format("cp45_slowneutral"), 30.0,  20, 630, 1),  #653

        ("{0}/{0}.wav".format("eg1_neutral"), 30.0,  25, 300, 0),  #320
        ("{0}/{0}.wav".format("eg2_neutral"), 30.0,  20, 387, 0),  #387
        ("{0}/{0}.wav".format("eg3_neutral"), 30.0,  30, 150, 0),  #195
        ("{0}/{0}.wav".format("eg4_neutral"), 30.0,  40, 277, 1),  #277
        ("{0}/{0}.wav".format("eg5_neutral"), 30.0,  30, 478, 1),  #479
        ("{0}/{0}.wav".format("eg6_neutral"), 30.0,  15, 265, 1),  #276
        ("{0}/{0}.wav".format("eg7_neutral"), 30.0,  40, 534, 0),  #534
        ("{0}/{0}.wav".format("eg8_neutral"), 30.0,  40, 898, 1),  #898
        ("{0}/{0}.wav".format("eg9_neutral"), 30.0,  25, 360, 1),  #380
        # # # ("{0}/{0}.wav".format("improv_joy"), 30.0,  0, 359),
        ("{0}/{0}.wav".format("mute_neutral"), 30.0,  0, 120, 0),      # do they point the correct frame given clipped range?
        ("{0}/{0}.wav".format("mute_amazement"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_joy"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_cheekiness"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_sadness"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_disgust"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_anger"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_fear"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_grief"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_pain"), 30.0,  0, 120, 0),
        ("{0}/{0}.wav".format("mute_outofbreath"), 30.0,  0, 120, 0),

        #('lindyBangkok/lindyBangkok.wav', 30.0,  0, 351, 0), #351

        # # add some male Chinese/English voice.
        # ("{0}/{0}.wav".format("chinese_tts_male_test_01_neutral"), 30.0,  0, 783),
        # ("{0}/{0}.wav".format("english_voice_male_p3_fear_neutral"), 30.0,  0, 315),
        # ("{0}/{0}.wav".format("english_voice_male_p3_neutral_neutral"), 30.0,  0, 281),
        # ("{0}/{0}.wav".format("english_voice_male_g8b_neutral"), 30.0,  0, 310),
        # ("{0}/{0}.wav".format("ljs_0_neutral"), 30.0,  0, 243),

        # ("{0}/{0}.wav".format("cp1_neutral_FEM"), 30.0,  25, 330, 1),  #349
        # ("{0}/{0}.wav".format("cp2_neutral_FEM"), 30.0,  20, 320, 1),  #344
        # ("{0}/{0}.wav".format("cp3_neutral_FEM"), 30.0,  15, 335, 1),  #338
        # ("{0}/{0}.wav".format("cp4_neutral_FEM"), 30.0,  25, 335, 1),  #356
        # ("{0}/{0}.wav".format("cp5_neutral_FEM"), 30.0,  30, 300, 1),  #320
        # ("{0}/{0}.wav".format("cp6_neutral_FEM"), 30.0,  30, 250, 4),  #284
        # ("{0}/{0}.wav".format("cp7_neutral_FEM"), 30.0,  20, 245, 0),  #251
        # ("{0}/{0}.wav".format("cp8_neutral_FEM"), 30.0,  25, 340, 1),  #350
        # ("{0}/{0}.wav".format("cp9_neutral_FEM"), 30.0,  35, 385, 1),  #397
        # ("{0}/{0}.wav".format("cp10_neutral_FEM"), 30.0,  30, 315, 0),  #319
        # ("{0}/{0}.wav".format("cp11_neutral_FEM"), 30.0,  20, 319, 1),  #319
        # ("{0}/{0}.wav".format("cp12_neutral_FEM"), 30.0,  25, 333, 1),  #333
        # ("{0}/{0}.wav".format("cp13_neutral_FEM"), 30.0,  20, 290, 1),  #296
        # ("{0}/{0}.wav".format("cp14_neutral_FEM"), 30.0,  15, 280, 1),  #303
        # ("{0}/{0}.wav".format("cp15_neutral_FEM"), 30.0,  25, 360, 0),  #403
        # ("{0}/{0}.wav".format("cp16_neutral_FEM"), 30.0,  35, 390, 1),  #390
        # ("{0}/{0}.wav".format("cp17_neutral_FEM"), 30.0,  40, 330, 0),  #373
        # ("{0}/{0}.wav".format("cp18_neutral_FEM"), 30.0,  25, 270, 0),  #280
        # ("{0}/{0}.wav".format("cp19_neutral_FEM"), 30.0,  60, 255, 0),  #281
        # ("{0}/{0}.wav".format("cp20_neutral_FEM"), 30.0,  30, 280, 0),  #289
        # ("{0}/{0}.wav".format("cp21_neutral_FEM"), 30.0,  30, 505, 0),  #551
        # ("{0}/{0}.wav".format("cp22_amazement_FEM"), 30.0,  25, 270, 0),  #281
        # ("{0}/{0}.wav".format("cp23_amazement_FEM"), 30.0,  20, 480, 1),  #527
        # ("{0}/{0}.wav".format("cp24_joy_FEM"), 30.0,  30, 273, 1),  #273
        # ("{0}/{0}.wav".format("cp25_joy_FEM"), 30.0,  25, 455, 1),  #495
        # ("{0}/{0}.wav".format("cp26_cheekiness_FEM"), 30.0,  20, 280, 1),  #304
        # ("{0}/{0}.wav".format("cp27_cheekiness_FEM"), 30.0,  40, 540, 1),  #575
        # ("{0}/{0}.wav".format("cp28_sadness_FEM"), 30.0,  40, 270, 1),  #273
        # ("{0}/{0}.wav".format("cp29_sadness_FEM"), 30.0,  45, 650, 1),  #711
        # ("{0}/{0}.wav".format("cp30_disgust_FEM"), 30.0,  20, 290, 1),  #317
        # ("{0}/{0}.wav".format("cp31_disgust_FEM"), 30.0,  25, 525, 1),  #682
        # ("{0}/{0}.wav".format("cp32_anger_FEM"), 30.0,  30, 235, 0),  #239
        # ("{0}/{0}.wav".format("cp33_anger_FEM"), 30.0,  25, 400, 1),  #504
        # ("{0}/{0}.wav".format("cp34_fear_FEM"), 30.0,  20, 280, 1),  #287
        # ("{0}/{0}.wav".format("cp35_fear_FEM"), 30.0,  30, 540, 1),  #596
        # ("{0}/{0}.wav".format("cp36_grief_FEM"), 30.0,  45, 370, 1),  #371
        # ("{0}/{0}.wav".format("cp37_grief_FEM"), 30.0,  35, 630, 1),  #661
        # ("{0}/{0}.wav".format("cp38_pain_FEM"), 30.0,  40, 285, 0),  #297
        # ("{0}/{0}.wav".format("cp39_pain_FEM"), 30.0,  30, 490, 0),  #549
        # ("{0}/{0}.wav".format("cp40_outofbreath_FEM"), 30.0,  45, 270, 0),  #293
        # ("{0}/{0}.wav".format("cp41_outofbreath_FEM"), 30.0,  30, 470, 1),  #504
        # ("{0}/{0}.wav".format("cp42_fastneutral_FEM"), 30.0,  15, 132, 0),  #132
        # ("{0}/{0}.wav".format("cp43_fastneutral_FEM"), 30.0,  20, 250, 1),  #271
        # ("{0}/{0}.wav".format("cp44_slowneutral_FEM"), 30.0,  30, 371, 1),  #371
        # ("{0}/{0}.wav".format("cp45_slowneutral_FEM"), 30.0,  20, 630, 1),  #653
        ("{0}/{0}.wav".format("eg1_neutral_FEM"), 30.0,  25, 300, 0),  #320
        ("{0}/{0}.wav".format("eg2_neutral_FEM"), 30.0,  20, 387, 0),  #387
        ("{0}/{0}.wav".format("eg3_neutral_FEM"), 30.0,  30, 150, 0),  #195
        ("{0}/{0}.wav".format("eg4_neutral_FEM"), 30.0,  40, 277, 1),  #277
        ("{0}/{0}.wav".format("eg5_neutral_FEM"), 30.0,  30, 478, 1),  #479
        ("{0}/{0}.wav".format("eg6_neutral_FEM"), 30.0,  15, 265, 1),  #276
        ("{0}/{0}.wav".format("eg7_neutral_FEM"), 30.0,  40, 534, 0),  #534
        ("{0}/{0}.wav".format("eg8_neutral_FEM"), 30.0,  40, 898, 1),  #898
        ("{0}/{0}.wav".format("eg9_neutral_FEM"), 30.0,  25, 360, 1),  #380


        # ("{0}/{0}.wav".format("cp1_neutral_MAL"), 30.0,  25, 330, 1),  #349
        # ("{0}/{0}.wav".format("cp2_neutral_MAL"), 30.0,  20, 320, 1),  #344
        # ("{0}/{0}.wav".format("cp3_neutral_MAL"), 30.0,  15, 335, 1),  #338
        # ("{0}/{0}.wav".format("cp4_neutral_MAL"), 30.0,  25, 335, 1),  #356
        # ("{0}/{0}.wav".format("cp5_neutral_MAL"), 30.0,  30, 300, 1),  #320
        # ("{0}/{0}.wav".format("cp6_neutral_MAL"), 30.0,  30, 250, 4),  #284
        # ("{0}/{0}.wav".format("cp7_neutral_MAL"), 30.0,  20, 245, 0),  #251
        # ("{0}/{0}.wav".format("cp8_neutral_MAL"), 30.0,  25, 340, 1),  #350
        # ("{0}/{0}.wav".format("cp9_neutral_MAL"), 30.0,  35, 385, 1),  #397
        # ("{0}/{0}.wav".format("cp10_neutral_MAL"), 30.0,  30, 315, 0),  #319
        # ("{0}/{0}.wav".format("cp11_neutral_MAL"), 30.0,  20, 319, 1),  #319
        # ("{0}/{0}.wav".format("cp12_neutral_MAL"), 30.0,  25, 333, 1),  #333
        # ("{0}/{0}.wav".format("cp13_neutral_MAL"), 30.0,  20, 290, 1),  #296
        # ("{0}/{0}.wav".format("cp14_neutral_MAL"), 30.0,  15, 280, 1),  #303
        # ("{0}/{0}.wav".format("cp15_neutral_MAL"), 30.0,  25, 360, 0),  #403
        # ("{0}/{0}.wav".format("cp16_neutral_MAL"), 30.0,  35, 390, 1),  #390
        # ("{0}/{0}.wav".format("cp17_neutral_MAL"), 30.0,  40, 330, 0),  #373
        # ("{0}/{0}.wav".format("cp18_neutral_MAL"), 30.0,  25, 270, 0),  #280
        # ("{0}/{0}.wav".format("cp19_neutral_MAL"), 30.0,  60, 255, 0),  #281
        # ("{0}/{0}.wav".format("cp20_neutral_MAL"), 30.0,  30, 280, 0),  #289
        # ("{0}/{0}.wav".format("cp21_neutral_MAL"), 30.0,  30, 505, 0),  #551
        # ("{0}/{0}.wav".format("cp22_amazement_MAL"), 30.0,  25, 270, 0),  #281
        # ("{0}/{0}.wav".format("cp23_amazement_MAL"), 30.0,  20, 480, 1),  #527
        # ("{0}/{0}.wav".format("cp24_joy_MAL"), 30.0,  30, 273, 1),  #273
        # ("{0}/{0}.wav".format("cp25_joy_MAL"), 30.0,  25, 455, 1),  #495
        # ("{0}/{0}.wav".format("cp26_cheekiness_MAL"), 30.0,  20, 280, 1),  #304
        # ("{0}/{0}.wav".format("cp27_cheekiness_MAL"), 30.0,  40, 540, 1),  #575
        # ("{0}/{0}.wav".format("cp28_sadness_MAL"), 30.0,  40, 270, 1),  #273
        # ("{0}/{0}.wav".format("cp29_sadness_MAL"), 30.0,  45, 650, 1),  #711
        # ("{0}/{0}.wav".format("cp30_disgust_MAL"), 30.0,  20, 290, 1),  #317
        # ("{0}/{0}.wav".format("cp31_disgust_MAL"), 30.0,  25, 525, 1),  #682
        # ("{0}/{0}.wav".format("cp32_anger_MAL"), 30.0,  30, 235, 0),  #239
        # ("{0}/{0}.wav".format("cp33_anger_MAL"), 30.0,  25, 400, 1),  #504
        # ("{0}/{0}.wav".format("cp34_fear_MAL"), 30.0,  20, 280, 1),  #287
        # ("{0}/{0}.wav".format("cp35_fear_MAL"), 30.0,  30, 540, 1),  #596
        # ("{0}/{0}.wav".format("cp36_grief_MAL"), 30.0,  45, 370, 1),  #371
        # ("{0}/{0}.wav".format("cp37_grief_MAL"), 30.0,  35, 630, 1),  #661
        # ("{0}/{0}.wav".format("cp38_pain_MAL"), 30.0,  40, 285, 0),  #297
        # ("{0}/{0}.wav".format("cp39_pain_MAL"), 30.0,  30, 490, 0),  #549
        # ("{0}/{0}.wav".format("cp40_outofbreath_MAL"), 30.0,  45, 270, 0),  #293
        # ("{0}/{0}.wav".format("cp41_outofbreath_MAL"), 30.0,  30, 470, 1),  #504
        # ("{0}/{0}.wav".format("cp42_fastneutral_MAL"), 30.0,  15, 132, 0),  #132
        # ("{0}/{0}.wav".format("cp43_fastneutral_MAL"), 30.0,  20, 250, 1),  #271
        # ("{0}/{0}.wav".format("cp44_slowneutral_MAL"), 30.0,  30, 371, 1),  #371
        # ("{0}/{0}.wav".format("cp45_slowneutral_MAL"), 30.0,  20, 630, 1),  #653
        ("{0}/{0}.wav".format("eg1_neutral_MAL"), 30.0,  25, 300, 0),  #320
        ("{0}/{0}.wav".format("eg2_neutral_MAL"), 30.0,  20, 387, 0),  #387
        ("{0}/{0}.wav".format("eg3_neutral_MAL"), 30.0,  30, 150, 0),  #195
        ("{0}/{0}.wav".format("eg4_neutral_MAL"), 30.0,  40, 277, 1),  #277
        ("{0}/{0}.wav".format("eg5_neutral_MAL"), 30.0,  30, 478, 1),  #479
        ("{0}/{0}.wav".format("eg6_neutral_MAL"), 30.0,  15, 265, 1),  #276
        ("{0}/{0}.wav".format("eg7_neutral_MAL"), 30.0,  40, 534, 0),  #534
        ("{0}/{0}.wav".format("eg8_neutral_MAL"), 30.0,  40, 898, 1),  #898
        ("{0}/{0}.wav".format("eg9_neutral_MAL"), 30.0,  25, 360, 1),  #380

    ],
}

dataset_extended_james = {
    'actor_mesh': '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/skin',
    'shots_root': '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/audio',
    'shots': [ # (shot_path, shot_fps, shot_start, shot_end)
        ("{0}/{0}.wav".format("ep1_neutral"), 30.0,  0, 200, 0),  #200
        ("{0}/{0}.wav".format("ep2_neutral"), 30.0,  0, 166, 0),  #166
        ("{0}/{0}.wav".format("ep3_neutral"), 30.0,  10, 320, 0),  #320
        ("{0}/{0}.wav".format("ep4_neutral"), 30.0,  0, 364, 0),  #364
        ("{0}/{0}.wav".format("ep1_amazement"), 30.0,  0, 215, 0),  #215
        ("{0}/{0}.wav".format("ep2_amazement"), 30.0,  0, 185, 0),  #232
        ("{0}/{0}.wav".format("ep3_amazement"), 30.0,  0, 382, 0),  #382
        ("{0}/{0}.wav".format("ep4_amazement"), 30.0,  15, 355, 0),  #385
        ("{0}/{0}.wav".format("ep1_joy"), 30.0,  0, 210, 0),  #235
        ("{0}/{0}.wav".format("ep2_joy"), 30.0,  0, 165, 0),  #186
        ("{0}/{0}.wav".format("ep3_joy"), 30.0,  0, 355, 0),  #403
        ("{0}/{0}.wav".format("ep4_joy"), 30.0,  8, 330, 0),  #348
        ("{0}/{0}.wav".format("ep1_cheekiness"), 30.0,  0, 220, 0),  #245
        ("{0}/{0}.wav".format("ep2_cheekiness"), 30.0,  0, 190, 0),  #215
        ("{0}/{0}.wav".format("ep3_cheekiness"), 30.0,  0, 370, 0),  #403
        ("{0}/{0}.wav".format("ep4_cheekiness"), 30.0,  30, 360, 0),  #376
        ("{0}/{0}.wav".format("ep1_sadness"), 30.0,  20, 280, 0),  #345
        ("{0}/{0}.wav".format("ep2_sadness"), 30.0,  20, 225, 0),  #256
        ("{0}/{0}.wav".format("ep3_sadness"), 30.0,  40, 440, 0),  #480
        ("{0}/{0}.wav".format("ep4_sadness"), 30.0,  10, 410, 0),  #455
        ("{0}/{0}.wav".format("ep1_disgust"), 30.0,  0, 231, 0),  #258
        ("{0}/{0}.wav".format("ep2_disgust"), 30.0,  20, 230, 0),  #282
        ("{0}/{0}.wav".format("ep3_disgust"), 30.0,  20, 390, 0),  #422
        ("{0}/{0}.wav".format("ep4_disgust"), 30.0,  20, 385, 0),  #431
        ("{0}/{0}.wav".format("ep1_anger"), 30.0,  0, 210, 0),  #263
        ("{0}/{0}.wav".format("ep2_anger"), 30.0,  0, 165, 0),  #196
        ("{0}/{0}.wav".format("ep3_anger"), 30.0,  0, 350, 0),  #370
        ("{0}/{0}.wav".format("ep4_anger"), 30.0,  10, 392, 0),  #392
        ("{0}/{0}.wav".format("ep1_fear"), 30.0,  10, 270, 0),  #335
        ("{0}/{0}.wav".format("ep2_fear"), 30.0,  10, 215, 0),  #284
        ("{0}/{0}.wav".format("ep3_fear"), 30.0,  10, 450, 0),  #513
        ("{0}/{0}.wav".format("ep4_fear"), 30.0,  10, 410, 0),  #451
        ("{0}/{0}.wav".format("ep1_grief"), 30.0,  20, 325, 0),  #341
        ("{0}/{0}.wav".format("ep2_grief"), 30.0,  50, 315, 0),  #359
        ("{0}/{0}.wav".format("ep3_grief"), 30.0,  30, 540, 0),  #553
        ("{0}/{0}.wav".format("ep4_grief"), 30.0,  0, 535, 0),  #576
        ("{0}/{0}.wav".format("ep1_pain"), 30.0,  0, 245, 0),  #289
        ("{0}/{0}.wav".format("ep2_pain"), 30.0,  0, 190, 0),  #241
        ("{0}/{0}.wav".format("ep3_pain"), 30.0,  10, 360, 0),  #388
        ("{0}/{0}.wav".format("ep4_pain"), 30.0,  10, 370, 0),  #425
        ("{0}/{0}.wav".format("ep1_outofbreath"), 30.0,  20, 350, 0),  #398
        ("{0}/{0}.wav".format("ep2_outofbreath"), 30.0,  50, 450, 0),  #490
        ("{0}/{0}.wav".format("ep3a_outofbreath"), 30.0,  40, 313, 0),  #313
        ("{0}/{0}.wav".format("ep3b_outofbreath"), 30.0,  0, 358, 0),  #358
        ("{0}/{0}.wav".format("ep3c_outofbreath"), 30.0,  30, 230, 0),  #251
        ("{0}/{0}.wav".format("ep4a_outofbreath"), 30.0,  0, 199, 0),  #199
        ("{0}/{0}.wav".format("ep4b_outofbreath"), 30.0,  40, 290, 0),  #319
        ("{0}/{0}.wav".format("ep4c_outofbreath"), 30.0,  0, 410, 0),  #445
        ("{0}/{0}.wav".format("eg1_neutral"), 30.0,  10, 330, 0),  #339
        ("{0}/{0}.wav".format("eg2_neutral"), 30.0,  0, 420, 0),  #428
        ("{0}/{0}.wav".format("eg3_neutral"), 30.0,  0, 169, 0),  #169
        ("{0}/{0}.wav".format("eg4_neutral"), 30.0,  0, 293, 0),  #293
        ("{0}/{0}.wav".format("eg5_neutral"), 30.0,  0, 558, 0),  #558
        ("{0}/{0}.wav".format("eg6_neutral"), 30.0,  25, 293, 0),  #293
        ("{0}/{0}.wav".format("eg7_neutral"), 30.0,  0, 505, 0),  #526
        ("{0}/{0}.wav".format("eg8a_neutral"), 30.0,  0, 276, 0),  #276
        ("{0}/{0}.wav".format("eg8b_neutral"), 30.0,  0, 351, 0),  #351
        ("{0}/{0}.wav".format("eg8c_neutral"), 30.0,  0, 255, 0),  #262
        ("{0}/{0}.wav".format("eg9_neutral"), 30.0,  10, 453, 0),  #453
        # ("{0}/{0}.wav".format("eg1_neutral"), 30.0,  0, 223, 0),  #223
        # ("{0}/{0}.wav".format("eg2_neutral"), 30.0,  0, 223, 0),  #223
        # ("{0}/{0}.wav".format("eg3_neutral"), 30.0,  0, 223, 0),  #223
        ("{0}/{0}.wav".format("ep2_fastneutral"), 30.0,  10, 140, 0),  #158
        ("{0}/{0}.wav".format("eg8_fastneutral"), 30.0,  10, 420, 0),  #440
        ("{0}/{0}.wav".format("eg9_fastneutral"), 30.0,  10, 230, 0),  #259
        ("{0}/{0}.wav".format("ep2_slowneutral"), 30.0,  0, 365, 0),  #386
        ("{0}/{0}.wav".format("eg8a_slowneutral"), 30.0,  0, 411, 0),  #411
        ("{0}/{0}.wav".format("eg8b_slowneutral"), 30.0,  0, 412, 0),  #412
        ("{0}/{0}.wav".format("eg8c_slowneutral"), 30.0,  0, 335, 0),  #335
        ("{0}/{0}.wav".format("eg9_slowneutral"), 30.0,  0, 565, 0),  #595

        ("{0}/{0}.wav".format("mute_neutral"), 30.0,  0, 119, 0),      # do they point the correct frame given clipped range?
        ("{0}/{0}.wav".format("mute_amazement"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_joy"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_cheekiness"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_sadness"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_disgust"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_anger"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_fear"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_grief"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_pain"), 30.0,  0, 119, 0),
        ("{0}/{0}.wav".format("mute_outofbreath"), 30.0,  0, 119, 0),


        ("{0}/{0}.wav".format("ep1_neutral_FEM"), 30.0,  0, 200, 0),  #200
        ("{0}/{0}.wav".format("ep2_neutral_FEM"), 30.0,  0, 166, 0),  #166
        ("{0}/{0}.wav".format("ep3_neutral_FEM"), 30.0,  10, 320, 0),  #320
        ("{0}/{0}.wav".format("ep4_neutral_FEM"), 30.0,  0, 364, 0),  #364
        ("{0}/{0}.wav".format("ep1_amazement_FEM"), 30.0,  0, 215, 0),  #215
        ("{0}/{0}.wav".format("ep2_amazement_FEM"), 30.0,  0, 185, 0),  #232
        ("{0}/{0}.wav".format("ep3_amazement_FEM"), 30.0,  0, 382, 0),  #382
        ("{0}/{0}.wav".format("ep4_amazement_FEM"), 30.0,  15, 355, 0),  #385
        ("{0}/{0}.wav".format("ep1_joy_FEM"), 30.0,  0, 210, 0),  #235
        ("{0}/{0}.wav".format("ep2_joy_FEM"), 30.0,  0, 165, 0),  #186
        ("{0}/{0}.wav".format("ep3_joy_FEM"), 30.0,  0, 355, 0),  #403
        ("{0}/{0}.wav".format("ep4_joy_FEM"), 30.0,  8, 330, 0),  #348
        ("{0}/{0}.wav".format("ep1_cheekiness_FEM"), 30.0,  0, 220, 0),  #245
        ("{0}/{0}.wav".format("ep2_cheekiness_FEM"), 30.0,  0, 190, 0),  #215
        ("{0}/{0}.wav".format("ep3_cheekiness_FEM"), 30.0,  0, 370, 0),  #403
        ("{0}/{0}.wav".format("ep4_cheekiness_FEM"), 30.0,  30, 360, 0),  #376
        ("{0}/{0}.wav".format("ep1_sadness_FEM"), 30.0,  20, 280, 0),  #345
        ("{0}/{0}.wav".format("ep2_sadness_FEM"), 30.0,  20, 225, 0),  #256
        ("{0}/{0}.wav".format("ep3_sadness_FEM"), 30.0,  40, 440, 0),  #480
        ("{0}/{0}.wav".format("ep4_sadness_FEM"), 30.0,  10, 410, 0),  #455
        ("{0}/{0}.wav".format("ep1_disgust_FEM"), 30.0,  0, 231, 0),  #258
        ("{0}/{0}.wav".format("ep2_disgust_FEM"), 30.0,  20, 230, 0),  #282
        ("{0}/{0}.wav".format("ep3_disgust_FEM"), 30.0,  20, 390, 0),  #422
        ("{0}/{0}.wav".format("ep4_disgust_FEM"), 30.0,  20, 385, 0),  #431
        ("{0}/{0}.wav".format("ep1_anger_FEM"), 30.0,  0, 210, 0),  #263
        ("{0}/{0}.wav".format("ep2_anger_FEM"), 30.0,  0, 165, 0),  #196
        ("{0}/{0}.wav".format("ep3_anger_FEM"), 30.0,  0, 350, 0),  #370
        ("{0}/{0}.wav".format("ep4_anger_FEM"), 30.0,  10, 392, 0),  #392
        ("{0}/{0}.wav".format("ep1_fear_FEM"), 30.0,  10, 270, 0),  #335
        ("{0}/{0}.wav".format("ep2_fear_FEM"), 30.0,  10, 215, 0),  #284
        ("{0}/{0}.wav".format("ep3_fear_FEM"), 30.0,  10, 450, 0),  #513
        ("{0}/{0}.wav".format("ep4_fear_FEM"), 30.0,  10, 410, 0),  #451
        ("{0}/{0}.wav".format("ep1_grief_FEM"), 30.0,  20, 325, 0),  #341
        ("{0}/{0}.wav".format("ep2_grief_FEM"), 30.0,  50, 315, 0),  #359
        ("{0}/{0}.wav".format("ep3_grief_FEM"), 30.0,  30, 540, 0),  #553
        ("{0}/{0}.wav".format("ep4_grief_FEM"), 30.0,  0, 535, 0),  #576
        ("{0}/{0}.wav".format("ep1_pain_FEM"), 30.0,  0, 245, 0),  #289
        ("{0}/{0}.wav".format("ep2_pain_FEM"), 30.0,  0, 190, 0),  #241
        ("{0}/{0}.wav".format("ep3_pain_FEM"), 30.0,  10, 360, 0),  #388
        ("{0}/{0}.wav".format("ep4_pain_FEM"), 30.0,  10, 370, 0),  #425
        ("{0}/{0}.wav".format("ep1_outofbreath_FEM"), 30.0,  20, 350, 0),  #398
        ("{0}/{0}.wav".format("ep2_outofbreath_FEM"), 30.0,  50, 450, 0),  #490
        ("{0}/{0}.wav".format("ep3a_outofbreath_FEM"), 30.0,  40, 313, 0),  #313
        ("{0}/{0}.wav".format("ep3b_outofbreath_FEM"), 30.0,  0, 358, 0),  #358
        ("{0}/{0}.wav".format("ep3c_outofbreath_FEM"), 30.0,  30, 230, 0),  #251
        ("{0}/{0}.wav".format("ep4a_outofbreath_FEM"), 30.0,  0, 199, 0),  #199
        ("{0}/{0}.wav".format("ep4b_outofbreath_FEM"), 30.0,  40, 290, 0),  #319
        ("{0}/{0}.wav".format("ep4c_outofbreath_FEM"), 30.0,  0, 410, 0),  #445
        # ("{0}/{0}.wav".format("eg1_neutral_FEM"), 30.0,  10, 330, 0),  #339
        # ("{0}/{0}.wav".format("eg2_neutral_FEM"), 30.0,  0, 420, 0),  #428
        # ("{0}/{0}.wav".format("eg3_neutral_FEM"), 30.0,  0, 169, 0),  #169
        # ("{0}/{0}.wav".format("eg4_neutral_FEM"), 30.0,  0, 293, 0),  #293
        # ("{0}/{0}.wav".format("eg5_neutral_FEM"), 30.0,  0, 558, 0),  #558
        # ("{0}/{0}.wav".format("eg6_neutral_FEM"), 30.0,  25, 293, 0),  #293
        # ("{0}/{0}.wav".format("eg7_neutral_FEM"), 30.0,  0, 505, 0),  #526
        # ("{0}/{0}.wav".format("eg8a_neutral_FEM"), 30.0,  0, 276, 0),  #276
        # ("{0}/{0}.wav".format("eg8b_neutral_FEM"), 30.0,  0, 351, 0),  #351
        # ("{0}/{0}.wav".format("eg8c_neutral_FEM"), 30.0,  0, 255, 0),  #262
        # ("{0}/{0}.wav".format("eg9_neutral_FEM"), 30.0,  10, 453, 0),  #453
        # # ("{0}/{0}.wav".format("eg1_neutral_FEM"), 30.0,  0, 223, 0),  #223
        # # ("{0}/{0}.wav".format("eg2_neutral_FEM"), 30.0,  0, 223, 0),  #223
        # # ("{0}/{0}.wav".format("eg3_neutral_FEM"), 30.0,  0, 223, 0),  #223
        # ("{0}/{0}.wav".format("ep2_fastneutral_FEM"), 30.0,  10, 140, 0),  #158
        # ("{0}/{0}.wav".format("eg8_fastneutral_FEM"), 30.0,  10, 420, 0),  #440
        # ("{0}/{0}.wav".format("eg9_fastneutral_FEM"), 30.0,  10, 230, 0),  #259
        # ("{0}/{0}.wav".format("ep2_slowneutral_FEM"), 30.0,  0, 365, 0),  #386
        # ("{0}/{0}.wav".format("eg8a_slowneutral_FEM"), 30.0,  0, 411, 0),  #411
        # ("{0}/{0}.wav".format("eg8b_slowneutral_FEM"), 30.0,  0, 412, 0),  #412
        # ("{0}/{0}.wav".format("eg8c_slowneutral_FEM"), 30.0,  0, 335, 0),  #335
        # ("{0}/{0}.wav".format("eg9_slowneutral_FEM"), 30.0,  0, 565, 0),  #595


        ("{0}/{0}.wav".format("ep1_neutral_MAL"), 30.0,  0, 200, 0),  #200
        ("{0}/{0}.wav".format("ep2_neutral_MAL"), 30.0,  0, 166, 0),  #166
        ("{0}/{0}.wav".format("ep3_neutral_MAL"), 30.0,  10, 320, 0),  #320
        ("{0}/{0}.wav".format("ep4_neutral_MAL"), 30.0,  0, 364, 0),  #364
        ("{0}/{0}.wav".format("ep1_amazement_MAL"), 30.0,  0, 215, 0),  #215
        ("{0}/{0}.wav".format("ep2_amazement_MAL"), 30.0,  0, 185, 0),  #232
        ("{0}/{0}.wav".format("ep3_amazement_MAL"), 30.0,  0, 382, 0),  #382
        ("{0}/{0}.wav".format("ep4_amazement_MAL"), 30.0,  15, 355, 0),  #385
        ("{0}/{0}.wav".format("ep1_joy_MAL"), 30.0,  0, 210, 0),  #235
        ("{0}/{0}.wav".format("ep2_joy_MAL"), 30.0,  0, 165, 0),  #186
        ("{0}/{0}.wav".format("ep3_joy_MAL"), 30.0,  0, 355, 0),  #403
        ("{0}/{0}.wav".format("ep4_joy_MAL"), 30.0,  8, 330, 0),  #348
        ("{0}/{0}.wav".format("ep1_cheekiness_MAL"), 30.0,  0, 220, 0),  #245
        ("{0}/{0}.wav".format("ep2_cheekiness_MAL"), 30.0,  0, 190, 0),  #215
        ("{0}/{0}.wav".format("ep3_cheekiness_MAL"), 30.0,  0, 370, 0),  #403
        ("{0}/{0}.wav".format("ep4_cheekiness_MAL"), 30.0,  30, 360, 0),  #376
        ("{0}/{0}.wav".format("ep1_sadness_MAL"), 30.0,  20, 280, 0),  #345
        ("{0}/{0}.wav".format("ep2_sadness_MAL"), 30.0,  20, 225, 0),  #256
        ("{0}/{0}.wav".format("ep3_sadness_MAL"), 30.0,  40, 440, 0),  #480
        ("{0}/{0}.wav".format("ep4_sadness_MAL"), 30.0,  10, 410, 0),  #455
        ("{0}/{0}.wav".format("ep1_disgust_MAL"), 30.0,  0, 231, 0),  #258
        ("{0}/{0}.wav".format("ep2_disgust_MAL"), 30.0,  20, 230, 0),  #282
        ("{0}/{0}.wav".format("ep3_disgust_MAL"), 30.0,  20, 390, 0),  #422
        ("{0}/{0}.wav".format("ep4_disgust_MAL"), 30.0,  20, 385, 0),  #431
        ("{0}/{0}.wav".format("ep1_anger_MAL"), 30.0,  0, 210, 0),  #263
        ("{0}/{0}.wav".format("ep2_anger_MAL"), 30.0,  0, 165, 0),  #196
        ("{0}/{0}.wav".format("ep3_anger_MAL"), 30.0,  0, 350, 0),  #370
        ("{0}/{0}.wav".format("ep4_anger_MAL"), 30.0,  10, 392, 0),  #392
        ("{0}/{0}.wav".format("ep1_fear_MAL"), 30.0,  10, 270, 0),  #335
        ("{0}/{0}.wav".format("ep2_fear_MAL"), 30.0,  10, 215, 0),  #284
        ("{0}/{0}.wav".format("ep3_fear_MAL"), 30.0,  10, 450, 0),  #513
        ("{0}/{0}.wav".format("ep4_fear_MAL"), 30.0,  10, 410, 0),  #451
        ("{0}/{0}.wav".format("ep1_grief_MAL"), 30.0,  20, 325, 0),  #341
        ("{0}/{0}.wav".format("ep2_grief_MAL"), 30.0,  50, 315, 0),  #359
        ("{0}/{0}.wav".format("ep3_grief_MAL"), 30.0,  30, 540, 0),  #553
        ("{0}/{0}.wav".format("ep4_grief_MAL"), 30.0,  0, 535, 0),  #576
        ("{0}/{0}.wav".format("ep1_pain_MAL"), 30.0,  0, 245, 0),  #289
        ("{0}/{0}.wav".format("ep2_pain_MAL"), 30.0,  0, 190, 0),  #241
        ("{0}/{0}.wav".format("ep3_pain_MAL"), 30.0,  10, 360, 0),  #388
        ("{0}/{0}.wav".format("ep4_pain_MAL"), 30.0,  10, 370, 0),  #425
        ("{0}/{0}.wav".format("ep1_outofbreath_MAL"), 30.0,  20, 350, 0),  #398
        ("{0}/{0}.wav".format("ep2_outofbreath_MAL"), 30.0,  50, 450, 0),  #490
        ("{0}/{0}.wav".format("ep3a_outofbreath_MAL"), 30.0,  40, 313, 0),  #313
        ("{0}/{0}.wav".format("ep3b_outofbreath_MAL"), 30.0,  0, 358, 0),  #358
        ("{0}/{0}.wav".format("ep3c_outofbreath_MAL"), 30.0,  30, 230, 0),  #251
        ("{0}/{0}.wav".format("ep4a_outofbreath_MAL"), 30.0,  0, 199, 0),  #199
        ("{0}/{0}.wav".format("ep4b_outofbreath_MAL"), 30.0,  40, 290, 0),  #319
        ("{0}/{0}.wav".format("ep4c_outofbreath_MAL"), 30.0,  0, 410, 0),  #445
        # ("{0}/{0}.wav".format("eg1_neutral_MAL"), 30.0,  10, 330, 0),  #339
        # ("{0}/{0}.wav".format("eg2_neutral_MAL"), 30.0,  0, 420, 0),  #428
        # ("{0}/{0}.wav".format("eg3_neutral_MAL"), 30.0,  0, 169, 0),  #169
        # ("{0}/{0}.wav".format("eg4_neutral_MAL"), 30.0,  0, 293, 0),  #293
        # ("{0}/{0}.wav".format("eg5_neutral_MAL"), 30.0,  0, 558, 0),  #558
        # ("{0}/{0}.wav".format("eg6_neutral_MAL"), 30.0,  25, 293, 0),  #293
        # ("{0}/{0}.wav".format("eg7_neutral_MAL"), 30.0,  0, 505, 0),  #526
        # ("{0}/{0}.wav".format("eg8a_neutral_MAL"), 30.0,  0, 276, 0),  #276
        # ("{0}/{0}.wav".format("eg8b_neutral_MAL"), 30.0,  0, 351, 0),  #351
        # ("{0}/{0}.wav".format("eg8c_neutral_MAL"), 30.0,  0, 255, 0),  #262
        # ("{0}/{0}.wav".format("eg9_neutral_MAL"), 30.0,  10, 453, 0),  #453
        # # ("{0}/{0}.wav".format("eg1_neutral_MAL"), 30.0,  0, 223, 0),  #223
        # # ("{0}/{0}.wav".format("eg2_neutral_MAL"), 30.0,  0, 223, 0),  #223
        # # ("{0}/{0}.wav".format("eg3_neutral_MAL"), 30.0,  0, 223, 0),  #223
        # ("{0}/{0}.wav".format("ep2_fastneutral_MAL"), 30.0,  10, 140, 0),  #158
        # ("{0}/{0}.wav".format("eg8_fastneutral_MAL"), 30.0,  10, 420, 0),  #440
        # ("{0}/{0}.wav".format("eg9_fastneutral_MAL"), 30.0,  10, 230, 0),  #259
        # ("{0}/{0}.wav".format("ep2_slowneutral_MAL"), 30.0,  0, 365, 0),  #386
        # ("{0}/{0}.wav".format("eg8a_slowneutral_MAL"), 30.0,  0, 411, 0),  #411
        # ("{0}/{0}.wav".format("eg8b_slowneutral_MAL"), 30.0,  0, 412, 0),  #412
        # ("{0}/{0}.wav".format("eg8c_slowneutral_MAL"), 30.0,  0, 335, 0),  #335
        # ("{0}/{0}.wav".format("eg9_slowneutral_MAL"), 30.0,  0, 565, 0),  #595
    ],
}


dataset_extended_mark = {
    #'actor_mesh': '/data/data/fa/mark/mesh/mark_mesh_hi',
    #'shots_root': '/data/data/fa/mark/all_hi/',
    #'actor_mesh': '/home/zyhuang/ssd/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/mesh/mark_mesh_hi',
    'shots_root': '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/all_hi/',
    'shots': [ # (shot_path, shot_fps, shot_start, shot_end, audio_offset) # original shot_end
        ('g1a/g1a.wav', 29.97,  0, 88, 0), #88
        ('g1b/g1b.wav', 29.97,  0, 84, 0), #84
        ('g1c/g1c.wav', 29.97,  0, 119, 0), #120
        ('g2a/g2a.wav', 29.97,  0, 113, 2), #113
        ('g2b/g2b.wav', 29.97,  0, 109, 0), #109
        ('g2c/g2c.wav', 29.97,  0, 140, 0), #140
        ('g3/g3.wav', 29.97,  0, 156, 0), #156
        ('g4a/g4a.wav', 29.97,  0, 128, 0), #128
        ('g4b/g4b.wav', 29.97,  0, 124, 0), #124
        ('g5a/g5a.wav', 29.97,  0, 144, 1), #144
        ('g5b/g5b.wav', 29.97,  0, 150, 0), #150
        ('g5c/g5c.wav', 29.97,  0, 126, 0), #126
        ('g6/g6.wav', 29.97,  0, 219, 0), #219   
        ('g7/g7.wav', 29.97,  0, 443, 0), #443
        ('g8a/g8a.wav', 29.97,  0, 251, 0), #251
        ('g8b/g8b.wav', 29.97,  0, 310, 0), #310
        ('g8c/g8c.wav', 29.97,  0, 225, 0), #225
        ('p1_neutral/p1_neutral.wav', 29.97,  0, 170, 0), #170
        ('p2_neutral/p2_neutral.wav', 29.97,  0, 218, 0), #218
        ('p3_neutral/p3_neutral.wav', 29.97,  0, 280, 0), #280
        ('p1_amazement/p1_amazement.wav', 29.97,  20, 215, 2), #235
        ('p2_amazement/p2_amazement.wav', 29.97,  0, 200, 1), #224
        ('p3_amazement/p3_amazement.wav', 29.97,  20, 340, 1), #355
        ('p1_anger/p1_anger.wav', 29.97,  10, 200, 3), #205
        ('p2_anger/p2_anger.wav', 29.97,  0, 205, 2), #214
        ('p3_anger/p3_anger.wav', 29.97,  0, 320, 2), #335
        ('p1_cheekiness/p1_cheekiness.wav', 29.97,  15, 200, 1), #213
        ('p2_cheekiness/p2_cheekiness.wav', 29.97,  20, 230, 0), #239 
        ('p3_cheekiness/p3_cheekiness.wav', 29.97,  20, 320, 1), #335
        ('p1_disgust/p1_disgust.wav', 29.97,  15, 220, 1), #223
        ('p2_disgust/p2_disgust.wav', 29.97,  15, 210, 0), #219
        ('p3_disgust/p3_disgust.wav', 29.97,  10, 325, 1), #331
        ('p1_fear/p1_fear.wav', 29.97,  15, 230, 1), #238
        ('p2_fear/p2_fear.wav', 29.97,  20, 210, 0), #235
        ('p3_fear/p3_fear.wav', 29.97,  0, 300, 1), #315
        ('p1_grief/p1_grief.wav', 29.97,  15, 220, 2), #231
        ('p2_grief/p2_grief.wav', 29.97,  10, 225, 1), #235
        ('p3_grief/p3_grief.wav', 29.97,  10, 380, 1), #393
        ('p1_joy/p1_joy.wav', 29.97,  20, 215, 1), #225
        ('p2_joy/p2_joy.wav', 29.97,  15, 210, 0), #238
        ('p3_joy/p3_joy.wav', 29.97,  15, 354, 1), #354
        ('p1_outofbreath/p1_outofbreath.wav', 29.97,  0, 201, 3), #201
        ('p2_outofbreath/p2_outofbreath.wav', 29.97,  0, 183, 2), #183
        ('p3_outofbreath/p3_outofbreath.wav', 29.97,  0, 310, 4), #313
        ('p1_pain/p1_pain.wav', 29.97,  15, 225, 2), #235
        ('p2_pain/p2_pain.wav', 29.97,  0, 205, 1), #209
        ('p3_pain/p3_pain.wav', 29.97,  0, 320, 1), #335
        ('p1_sadness/p1_sadness.wav', 29.97,  0, 224, 0), #224
        ('p2_sadness/p2_sadness.wav', 29.97,  0, 228, 0), #228
        ('p3_sadness/p3_sadness.wav', 29.97,  27, 355, 0), #366
        ("{0}/{0}.wav".format("mute_neutral"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_amazement"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_joy"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_cheekiness"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_sadness"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_disgust"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_anger"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_fear"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_grief"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_pain"), 30.0,  0, 119, 0), #120
        ("{0}/{0}.wav".format("mute_outofbreath"), 30.0,  0, 119, 0), #120

        #('lindyBangkok/lindyBangkok.wav', 29.97,  0, 351, 0), #351
        #('yseolMBP/yseolMBP.wav', 29.97,  0, 529, 0), #529
        #('yseolMBP_anger/yseolMBP_anger.wav', 29.97,  0, 529, 0), #529

        # ('g1a_BLE/g1a_BLE.wav', 29.97,  0, 88, 0), #88
        # ('g5a_BLE/g5a_BLE.wav', 29.97,  0, 144, 1), #144
        # ('g8b_BLE/g8b_BLE.wav', 29.97,  0, 310, 0), #310
        # ('p1_neutral_BLE/p1_neutral_BLE.wav', 29.97,  0, 170, 0), #170
        # ('p2_neutral_BLE/p2_neutral_BLE.wav', 29.97,  0, 218, 0), #218
        # ('p3_neutral_BLE/p3_neutral_BLE.wav', 29.97,  0, 280, 0), #280
        
        # conversion to RIVA female1 voice
        ('g1a_FEM/g1a_FEM.wav', 29.97,  0, 88, 0), #88
        ('g1b_FEM/g1b_FEM.wav', 29.97,  0, 84, 0), #84
        ('g1c_FEM/g1c_FEM.wav', 29.97,  0, 119, 0), #120
        ('g2a_FEM/g2a_FEM.wav', 29.97,  0, 113, 2), #113
        ('g2b_FEM/g2b_FEM.wav', 29.97,  0, 109, 0), #109
        ('g2c_FEM/g2c_FEM.wav', 29.97,  0, 140, 0), #140
        ('g3_FEM/g3_FEM.wav', 29.97,  0, 156, 0), #156
        ('g4a_FEM/g4a_FEM.wav', 29.97,  0, 128, 0), #128
        ('g4b_FEM/g4b_FEM.wav', 29.97,  0, 124, 0), #124
        ('g5a_FEM/g5a_FEM.wav', 29.97,  0, 144, 1), #144
        ('g5b_FEM/g5b_FEM.wav', 29.97,  0, 150, 0), #150
        ('g5c_FEM/g5c_FEM.wav', 29.97,  0, 126, 0), #126
        ('g6_FEM/g6_FEM.wav', 29.97,  0, 219, 0), #219   
        ('g7_FEM/g7_FEM.wav', 29.97,  0, 443, 0), #443
        ('g8a_FEM/g8a_FEM.wav', 29.97,  0, 251, 0), #251
        ('g8b_FEM/g8b_FEM.wav', 29.97,  0, 310, 0), #310
        ('g8c_FEM/g8c_FEM.wav', 29.97,  0, 225, 0), #225
        ('p1_neutral_FEM/p1_neutral_FEM.wav', 29.97,  0, 170, 0), #170
        ('p2_neutral_FEM/p2_neutral_FEM.wav', 29.97,  0, 218, 0), #218
        ('p3_neutral_FEM/p3_neutral_FEM.wav', 29.97,  0, 280, 0), #280
        # ('p1_amazement_FEM/p1_amazement_FEM.wav', 29.97,  20, 215, 2), #235
        # ('p2_amazement_FEM/p2_amazement_FEM.wav', 29.97,  0, 200, 1), #224
        # ('p3_amazement_FEM/p3_amazement_FEM.wav', 29.97,  20, 340, 1), #355
        # ('p1_anger_FEM/p1_anger_FEM.wav', 29.97,  10, 200, 3), #205
        # ('p2_anger_FEM/p2_anger_FEM.wav', 29.97,  0, 205, 2), #214
        # ('p3_anger_FEM/p3_anger_FEM.wav', 29.97,  0, 320, 2), #335
        # ('p1_cheekiness_FEM/p1_cheekiness_FEM.wav', 29.97,  15, 200, 1), #213
        # ('p2_cheekiness_FEM/p2_cheekiness_FEM.wav', 29.97,  20, 230, 0), #239 
        # ('p3_cheekiness_FEM/p3_cheekiness_FEM.wav', 29.97,  20, 320, 1), #335
        # ('p1_disgust_FEM/p1_disgust_FEM.wav', 29.97,  15, 220, 1), #223
        # ('p2_disgust_FEM/p2_disgust_FEM.wav', 29.97,  15, 210, 0), #219
        # ('p3_disgust_FEM/p3_disgust_FEM.wav', 29.97,  10, 325, 1), #331
        # ('p1_fear_FEM/p1_fear_FEM.wav', 29.97,  15, 230, 1), #238
        # ('p2_fear_FEM/p2_fear_FEM.wav', 29.97,  20, 210, 0), #235
        # ('p3_fear_FEM/p3_fear_FEM.wav', 29.97,  0, 300, 1), #315
        # ('p1_grief_FEM/p1_grief_FEM.wav', 29.97,  15, 220, 2), #231
        # ('p2_grief_FEM/p2_grief_FEM.wav', 29.97,  10, 225, 1), #235
        # ('p3_grief_FEM/p3_grief_FEM.wav', 29.97,  10, 380, 1), #393
        # ('p1_joy_FEM/p1_joy_FEM.wav', 29.97,  20, 215, 1), #225
        # ('p2_joy_FEM/p2_joy_FEM.wav', 29.97,  15, 210, 0), #238
        # ('p3_joy_FEM/p3_joy_FEM.wav', 29.97,  15, 354, 1), #354
        # ('p1_outofbreath_FEM/p1_outofbreath_FEM.wav', 29.97,  0, 201, 3), #201
        # ('p2_outofbreath_FEM/p2_outofbreath_FEM.wav', 29.97,  0, 183, 2), #183
        # ('p3_outofbreath_FEM/p3_outofbreath_FEM.wav', 29.97,  0, 310, 4), #313
        # ('p1_pain_FEM/p1_pain_FEM.wav', 29.97,  15, 225, 2), #235
        # ('p2_pain_FEM/p2_pain_FEM.wav', 29.97,  0, 205, 1), #209
        # ('p3_pain_FEM/p3_pain_FEM.wav', 29.97,  0, 320, 1), #335
        # ('p1_sadness_FEM/p1_sadness_FEM.wav', 29.97,  0, 224, 0), #224
        # ('p2_sadness_FEM/p2_sadness_FEM.wav', 29.97,  0, 228, 0), #228
        # ('p3_sadness_FEM/p3_sadness_FEM.wav', 29.97,  27, 355, 0), #366

        ('g1a_MAL/g1a_MAL.wav', 29.97,  0, 88, 0), #88
        ('g1b_MAL/g1b_MAL.wav', 29.97,  0, 84, 0), #84
        ('g1c_MAL/g1c_MAL.wav', 29.97,  0, 119, 0), #120
        ('g2a_MAL/g2a_MAL.wav', 29.97,  0, 113, 2), #113
        ('g2b_MAL/g2b_MAL.wav', 29.97,  0, 109, 0), #109
        ('g2c_MAL/g2c_MAL.wav', 29.97,  0, 140, 0), #140
        ('g3_MAL/g3_MAL.wav', 29.97,  0, 156, 0), #156
        ('g4a_MAL/g4a_MAL.wav', 29.97,  0, 128, 0), #128
        ('g4b_MAL/g4b_MAL.wav', 29.97,  0, 124, 0), #124
        ('g5a_MAL/g5a_MAL.wav', 29.97,  0, 144, 1), #144
        ('g5b_MAL/g5b_MAL.wav', 29.97,  0, 150, 0), #150
        ('g5c_MAL/g5c_MAL.wav', 29.97,  0, 126, 0), #126
        ('g6_MAL/g6_MAL.wav', 29.97,  0, 219, 0), #219   
        ('g7_MAL/g7_MAL.wav', 29.97,  0, 443, 0), #443
        ('g8a_MAL/g8a_MAL.wav', 29.97,  0, 251, 0), #251
        ('g8b_MAL/g8b_MAL.wav', 29.97,  0, 310, 0), #310
        ('g8c_MAL/g8c_MAL.wav', 29.97,  0, 225, 0), #225
        ('p1_neutral_MAL/p1_neutral_MAL.wav', 29.97,  0, 170, 0), #170
        ('p2_neutral_MAL/p2_neutral_MAL.wav', 29.97,  0, 218, 0), #218
        # ('p3_neutral_MAL/p3_neutral_MAL.wav', 29.97,  0, 280, 0), #280
        # ('p1_amazement_MAL/p1_amazement_MAL.wav', 29.97,  20, 215, 2), #235
        # ('p2_amazement_MAL/p2_amazement_MAL.wav', 29.97,  0, 200, 1), #224
        # ('p3_amazement_MAL/p3_amazement_MAL.wav', 29.97,  20, 340, 1), #355
        # ('p1_anger_MAL/p1_anger_MAL.wav', 29.97,  10, 200, 3), #205
        # ('p2_anger_MAL/p2_anger_MAL.wav', 29.97,  0, 205, 2), #214
        # ('p3_anger_MAL/p3_anger_MAL.wav', 29.97,  0, 320, 2), #335
        # ('p1_cheekiness_MAL/p1_cheekiness_MAL.wav', 29.97,  15, 200, 1), #213
        # ('p2_cheekiness_MAL/p2_cheekiness_MAL.wav', 29.97,  20, 230, 0), #239 
        # ('p3_cheekiness_MAL/p3_cheekiness_MAL.wav', 29.97,  20, 320, 1), #335
        # ('p1_disgust_MAL/p1_disgust_MAL.wav', 29.97,  15, 220, 1), #223
        # ('p2_disgust_MAL/p2_disgust_MAL.wav', 29.97,  15, 210, 0), #219
        # ('p3_disgust_MAL/p3_disgust_MAL.wav', 29.97,  10, 325, 1), #331
        # ('p1_fear_MAL/p1_fear_MAL.wav', 29.97,  15, 230, 1), #238
        # ('p2_fear_MAL/p2_fear_MAL.wav', 29.97,  20, 210, 0), #235
        # ('p3_fear_MAL/p3_fear_MAL.wav', 29.97,  0, 300, 1), #315
        # ('p1_grief_MAL/p1_grief_MAL.wav', 29.97,  15, 220, 2), #231
        # ('p2_grief_MAL/p2_grief_MAL.wav', 29.97,  10, 225, 1), #235
        # ('p3_grief_MAL/p3_grief_MAL.wav', 29.97,  10, 380, 1), #393
        # ('p1_joy_MAL/p1_joy_MAL.wav', 29.97,  20, 215, 1), #225
        # ('p2_joy_MAL/p2_joy_MAL.wav', 29.97,  15, 210, 0), #238
        # ('p3_joy_MAL/p3_joy_MAL.wav', 29.97,  15, 354, 1), #354
        # ('p1_outofbreath_MAL/p1_outofbreath_MAL.wav', 29.97,  0, 201, 3), #201
        # ('p2_outofbreath_MAL/p2_outofbreath_MAL.wav', 29.97,  0, 183, 2), #183
        # ('p3_outofbreath_MAL/p3_outofbreath_MAL.wav', 29.97,  0, 310, 4), #313
        # ('p1_pain_MAL/p1_pain_MAL.wav', 29.97,  15, 225, 2), #235
        # ('p2_pain_MAL/p2_pain_MAL.wav', 29.97,  0, 205, 1), #209
        # ('p3_pain_MAL/p3_pain_MAL.wav', 29.97,  0, 320, 1), #335
        # ('p1_sadness_MAL/p1_sadness_MAL.wav', 29.97,  0, 224, 0), #224
        # ('p2_sadness_MAL/p2_sadness_MAL.wav', 29.97,  0, 228, 0), #228
        # ('p3_sadness_MAL/p3_sadness_MAL.wav', 29.97,  27, 355, 0), #366
    ],
}


# claire
#pca_data_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/skin/skin_pca_data.npz' 
pca_data_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/skin_topo2/skin_pca_data.npz'
#pca_coeff_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/skin_topo2/skin_pca_coeffs_fixed_0611.npz' # lindyBangkok added
pca_coeff_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/skin_topo2/skin_pca_coeffs_training_shots_mutes_adjust.npz'
num_pca_basis_claire = 140

# mark
# pca_data_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/skin/skin_pca_data_ronanFix_0723.npz' 
# pca_coeff_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/skin/skin_pca_coeffs_ronanFixed_0723.npz'
pca_data_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/skin_topo2/skin_pca_data.npz'
pca_coeff_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/skin_topo2/skin_pca_coeffs_training_shots_mutes.npz'
num_pca_basis_mark = 140

# james
pca_data_path_james = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/skin_topo2/skin_pca_data_ronanFix0828.npz' 
pca_coeff_path_james = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/skin_topo2/skin_pca_coeffs_training_shots_mutes_adjust_ronanFix0828.npz'
num_pca_basis_james = 140


# pca data load
pca_data_claire = np.load(pca_data_path_claire)
pca_data_mark = np.load(pca_data_path_mark)
pca_data_james = np.load(pca_data_path_james)

 
# claire
evecs_t_claire = pca_data_claire['evecs_t'][:,:num_pca_basis_claire]   # 99% # 272 - 99.9%
mean_claire = pca_data_claire['mean']

# mark
evecs_t_mark = pca_data_mark['evecs_t'][:,:num_pca_basis_mark]   # 99% # 272 - 99.9%
mean_mark = pca_data_mark['mean']

# james
evecs_t_james = pca_data_james['evecs_t'][:,:num_pca_basis_james]   # 99% # 272 - 99.9%
mean_james = pca_data_james['mean']

# claire
#offset and scale estimated from mean, to make mean zero centered and have values roughly in range [-1, 1]
offset_claire = np.array([0.097, 142.447, -1.999]) 
scale_claire = 20
tongue_data_scale_claire = 70 #estimated from all tracks
gum_data_scale_claire = 10 # /10 to make it fluctuate in range [0,0.1]
eyeball_data_scale_claire = 1 # not using eyeball data for now

# mark 
#offset and scale estimated from mean, to make mean zero centered and have values roughly in range [-1, 1]
offset_mark = np.array([-0.109, 154.372, -1.077]) 
scale_mark = 20
tongue_data_scale_mark = 70 #estimated from all tracks
gum_data_scale_mark = 10 # /10 to make it fluctuate in range [0,0.1]
eyeball_data_scale_mark = 1 # not using eyeball data for now

# james
#offset and scale estimated from mean, to make mean zero centered and have values roughly in range [-1, 1]
offset_james = np.array([-0.023, 169.291, 9.892]) 
scale_james = 20
#TODO is this optimal??
tongue_data_scale_james = 70
gum_data_scale_james = 10
eyeball_data_scale_james = 1



# claire
pca_tongue_data_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/tongue/tongue_stab_neutral_pca_data.npz'
pca_tongue_coeff_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/tongue/tongue_stab_neutral_pca_coeff_fixed_0430.npz'
bottomGum_data_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/bottomGum/bottomGum_mean_shots_all_fixed_0430.npz'
eyeballs_rot_data_path_claire = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/dian/eyeball/eyeballRotData_all_fixed_0430.npz' # eyeball rotation adjusted for 

# mark
pca_tongue_data_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/tongue/tongue_stab_neutral_pca_data.npz'
pca_tongue_coeff_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/tongue/tongue_pca_coeff_fixed_0312.npz'
bottomGum_data_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/bottomGum/bottomGum_mean_shots_re0408_mute.npz'
eyeballs_rot_data_path_mark = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/mark/eyeball/eyeballRotData_all_emo_mutes_fixed_0312.npz' # eyeball rotation adjusted for 

# james
pca_tongue_data_path_james = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/tongue/tongue_pca_data.npz'
pca_tongue_coeff_path_james = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/tongue/tongue_pca_coeff_mutes.npz'
bottomGum_data_path_james = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/bottomGum/bottomGum_mean_shots_all_emo_mutes.npz'
eyeballs_rot_data_path_james = '/workspace/Shared_drives/Audio2Face-dev/data/a2f_training_data/edem/eyeball/eyeballRotData_all_emo_mutes.npz'


tongue_dim, gum_dim, eye_dim = 10, 15, 4

if __name__ == "__main__":

    templates = []
    templates_full = []
    for id in ['claire', 'mark', 'james']:

        if id == 'claire':
            mean = mean_claire
            offset = offset_claire
            scale = scale_claire
            pca_coeff_path = pca_coeff_path_claire
            pca_tongue_coeff_path = pca_tongue_coeff_path_claire
            bottomGum_data_path = bottomGum_data_path_claire
            eyeballs_rot_data_path = eyeballs_rot_data_path_claire
            dataset_extended = dataset_extended_claire
            evecs_t = evecs_t_claire
            tongue_data_scale = tongue_data_scale_claire
            gum_data_scale = gum_data_scale_claire
            eyeball_data_scale = eyeball_data_scale_claire
        elif id == 'mark':
            mean = mean_mark
            offset = offset_mark
            scale = scale_mark
            pca_coeff_path = pca_coeff_path_mark
            pca_tongue_coeff_path = pca_tongue_coeff_path_mark
            bottomGum_data_path = bottomGum_data_path_mark
            eyeballs_rot_data_path = eyeballs_rot_data_path_mark
            dataset_extended = dataset_extended_mark
            evecs_t = evecs_t_mark
            tongue_data_scale = tongue_data_scale_mark
            gum_data_scale = gum_data_scale_mark
            eyeball_data_scale = eyeball_data_scale_mark
        elif id == 'james':
            mean = mean_james
            offset = offset_james
            scale = scale_james
            pca_coeff_path = pca_coeff_path_james
            pca_tongue_coeff_path = pca_tongue_coeff_path_james
            bottomGum_data_path = bottomGum_data_path_james
            eyeballs_rot_data_path = eyeballs_rot_data_path_james
            dataset_extended = dataset_extended_james
            evecs_t = evecs_t_james
            tongue_data_scale = tongue_data_scale_james
            gum_data_scale = gum_data_scale_james
            eyeball_data_scale = eyeball_data_scale_james
        
        
        tongue_max = 0
        tongue_min = 99999
        
        gum_max = 0
        gum_min = 99999

        eyeball_max = 0
        eyeball_min = 99999

        os.makedirs(f"data/A2F/{id}", exist_ok=True)
        #save mean as template 
        with open(f'data/A2F/{id}/templates.pkl', 'wb') as file:
            template = (mean.reshape(-1,3)-offset)/scale
            templates.append(template)
            template = np.concatenate([template.reshape((-1)), np.zeros(tongue_dim+gum_dim+eye_dim)])
            templates_full.append(template)
            pickle.dump({f'{id}':template}, file)
            # pickle.dump({'A2F_Claire':mean.reshape(-1,3)}, file) # non scaled version
            print(template.min(axis=0))
            print(template.max(axis=0))
        target_reader = np.load(pca_coeff_path)
        target_reader_tongue = np.load(pca_tongue_coeff_path)
        target_reader_bottomGum = np.load(bottomGum_data_path)
        target_reader_eyeballs = np.load(eyeballs_rot_data_path)

        shots = read_shot_list(dataset_extended)

        os.makedirs(f"data/A2F/{id}/vertices_npy/", exist_ok=True)
        os.makedirs(f"data/A2F/{id}/wav/", exist_ok=True)

        tongue_mean_list = []
        bottomGum_mean_list = []
        eyeballs_mean_list = []

        all_target_name_read = []
        all_tongue_data = []
        all_bottomGum_data = []
        all_eyeballs_data = []

        for shot in shots:
            #print(shot)
            target_fpath = os.path.join(shot.data_root, shot.target_fpath)
            # target_data = self.target_reader(target_fpath)
            target_name = shot.target_fpath.split('/')[0]

            if '_BLE' in target_name:
                target_name_read = target_name[:-4]
            elif '_MAL' in target_name:
                target_name_read = target_name[:-4]
            elif '_FEM' in target_name:
                target_name_read = target_name[:-4]
            else:
                target_name_read = target_name

            target_data = target_reader[target_name_read].T # nFrame, 140 (nPCA)
            target_data_tongue = target_reader_tongue[target_name_read].T #nFrame, 10
            target_data_bottomGum = target_reader_bottomGum[target_name_read]

            target_data_bottomGum = target_data_bottomGum.reshape(target_data_bottomGum.shape[0], -1) #nFrame, 15
            target_data_eyeballs = target_reader_eyeballs[target_name_read]# nFrame, 4
            # print(f"read {target_name_read} from {target_name}", target_data.shape, target_data_tongue.shape, target_data_bottomGum.shape, target_data_eyeballs.shape)
            minLength = min([target_data.shape[0], target_data_tongue.shape[0], target_data_bottomGum.shape[0], target_data_eyeballs.shape[0]])
            #print(minLength, target_data.shape[0])
            #assert minLength == target_data.shape[0]
            reconstructed_mesh = np.dot(target_data, evecs_t.T) + mean #reconstruct vertex position from pca
            #save reconstructed_mesh as npy using np.save,  as shape nFrame, 3V, where V is the vertex number, with name, A2F_Claire_${target_name}.npy
            output_mesh_path = f'data/A2F/{id}/vertices_npy/{target_name}.npy'
            N = reconstructed_mesh.shape[0]
            reconstructed_mesh = ((reconstructed_mesh.reshape(N, -1, 3) - offset) / scale).reshape(N,-1) # apply offset and scale to gt cache as well.
            

            tongue_mean, bottomGum_mean, eyeballs_mean = np.mean(target_data_tongue, axis=0), np.mean(target_data_bottomGum, axis=0), np.mean(target_data_eyeballs, axis=0)
            # target_data_tongue_normalized, tongue_mean, tongue_std = normalize_data(target_data_tongue, mean=tongue_mean, std=tongue_std)
            # target_data_bottomGum_normalized, bottomGum_mean, bottomGum_std = normalize_data(target_data_bottomGum, mean=bottomGum_mean, std=bottomGum_std)
            # target_data_eyeballs_normalized, eyeballs_mean, eyeballs_std = normalize_data(target_data_eyeballs, mean=eyeballs_mean, std=eyeballs_std)
            # print('mean:', bottomGum_mean)
            # print('std',  bottomGum_std)
            # print(np.abs(target_data_tongue_normalized).max(), np.abs(target_data_bottomGum_normalized).max(), np.abs(target_data_eyeballs_normalized).max())
            # assert target_data_tongue_normalized.max() < 100
            # assert target_data_bottomGum_normalized.max() < 100
            # assert target_data_eyeballs_normalized.max() < 100
            tongue_mean_list.append(tongue_mean)
            bottomGum_mean_list.append(bottomGum_mean)
            eyeballs_mean_list.append(eyeballs_mean)

            all_target_name_read.append(target_name_read)
            all_tongue_data.append(target_data_tongue[:minLength])
            all_bottomGum_data.append(target_data_bottomGum[:minLength])
            all_eyeballs_data.append(target_data_eyeballs[:minLength])

            #if id == 'james':
            target_data_bottomGum[:,[[0, 3, 6, 9, 12]]] = 0 # enforce x of gum kp to be zero. I found 3 synthetic cache data has non-zero x pos, but all other tracks are 0.
            target_data_eyeballs = np.zeros_like(target_data_eyeballs) # don't use eyeball data for now
            #concatename all output (184589), in order face vert(184560), tongue(10), gum(15), eye(4).
            ###
            tongue_max = max(target_data_tongue.max(), tongue_max)
            tongue_min = min(target_data_tongue.min(), tongue_min)
            gum_max = max(target_data_bottomGum.max(), gum_max)
            gum_min = min(target_data_bottomGum.min(), gum_min)
            eyeball_max = max(target_data_eyeballs.max(), eyeball_max)
            eyeball_min = min(target_data_eyeballs.min(), eyeball_min)

            ###

            np.save(output_mesh_path, np.concatenate([reconstructed_mesh[:minLength], target_data_tongue[:minLength]/tongue_data_scale, target_data_bottomGum[:minLength]/gum_data_scale, target_data_eyeballs[:minLength]/eyeball_data_scale], axis=1))

            #cp the audio to the wav folder, with name A2F_Claire_${target_name}.wav
            audio_fpath = os.path.join(shot.data_root, shot.audio_fpath)
            output_audio_path = f'data/A2F/{id}/wav/{target_name}.wav'
            shutil.copy(audio_fpath, output_audio_path)
        
        tongue_mean = np.mean(tongue_mean_list, axis=0)
        bottomGum_mean = np.mean(bottomGum_mean_list, axis=0)
        eyeballs_mean = np.mean(eyeballs_mean_list, axis=0)

        all_target_name_read, all_tongue_data,all_bottomGum_data, all_eyeballs_data =  np.array(all_target_name_read), np.concatenate(all_tongue_data, axis=0),  np.concatenate(all_bottomGum_data, axis=0), np.concatenate(all_eyeballs_data, axis=0)
        np.save(f'data/A2F/{id}/target_name_read.npy', np.array(all_target_name_read))
        np.save(f'data/A2F/{id}/tongue_data.npy', np.concatenate(all_tongue_data, axis=0))
        np.save(f'data/A2F/{id}/bottomGum_data.npy', np.concatenate(all_bottomGum_data, axis=0))
        np.save(f'data/A2F/{id}/eyeballs_data.npy', np.concatenate(all_eyeballs_data, axis=0))

        scaling_data = {
        'tongue_mean': tongue_mean,
        'bottomGum_mean': bottomGum_mean,
        'eyeballs_mean': eyeballs_mean,
        'tongue_data_scale':tongue_data_scale,
        'gum_data_scale':gum_data_scale, #all_bottomGum_data.max() = 0.31
        'eye_data_scale': 1 #all_eyeballs_data.max() = 20.7! which is too large, skip using eye data for now.
        }
        print("saving scaling stats:", scaling_data)
        np.save(f'data/A2F/{id}/scaling_data.npy', scaling_data)
        print(id)
        print(tongue_max, tongue_min, gum_max, gum_min, eyeball_max, eyeball_min)

    templates_full_mean = np.mean(np.stack(templates_full, axis=0), axis=0)
    print('templates_full_mean', templates_full_mean.shape)
    with open(f'data/A2F/templates_full_mean.pkl', 'wb') as file:
        pickle.dump({'templates_full_mean':templates_full_mean}, file)

    # Calculate mean across all templates
    mean_template = np.mean(np.stack(templates, axis=0), axis=0)
    print(mean_template.shape)
    
    # Read original template obj file
    template_vertices = []
    template_faces = []
    with open('data/A2F/face_template.obj', 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Skip vertex lines as we'll replace them
                continue
            elif line.startswith('f '):
                template_faces.append(line)
                
    # Write new obj with mean template vertices
    with open('data/A2F/face_template_mean.obj', 'w') as f:
        # Write vertices
        for v in mean_template:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        # Write faces
        for face in template_faces:
            f.write(face)
    
    '''
    # Save as .pc file for meshlab visualization
    output_pc = os.path.join('data/A2F', f'mean_template.pc')
    with open(output_pc, 'w') as f:
        for vert in mean_template:
            f.write(f'{vert[0]} {vert[1]} {vert[2]}\n')
    '''
