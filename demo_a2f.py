import os
import pickle
import torch
import torch.nn.functional as F

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate

import numpy as np
# blink_exp_betas = np.array(
#     [0.04676158497927314, 0.03758675711005459, -0.8504121184951298, 0.10082324210507627, -0.574142329926028,
#         0.6440016589938355, 0.36403779939335984, 0.21642312586261656, 0.6754551784690193, 1.80958618462892,
#         0.7790133813372259, -0.24181691256476057, 0.826280685961679, -0.013525679499256753, 1.849393698014113,
#         -0.263035686247264, 0.42284248271332153, 0.10550891351425384, 0.6720993875023772, 0.41703592560736436,
#         3.308019065485072, 1.3358509602858895, 1.2997143108969278, -1.2463587328652894, -1.4818961382824924,
#         -0.6233880069345369, 0.26812528424728455, 0.5154889093160832, 0.6116267181402183, 0.9068826814583771,
#         -0.38869613253448576, 1.3311776710005476, -0.5802565274559162, -0.7920775624092143, -1.3278601781150017,
#         -1.2066425872386706, 0.34250140710360893, -0.7230686724732668, -0.6859285483325263, -1.524877347586566,
#         -1.2639479212965923, -0.019294228307535275, 0.2906175769381998, -1.4082782880837976, 0.9095436721066045,
#         1.6007365724960054, 2.0302381182163574, 0.5367600947801505, -0.12233184771794232, -0.506024823810769,
#         2.4312326730634783, 0.5622323258974669, 0.19022395712837198, -0.7729758559103581, -1.5624233513002923,
#         0.8275863297957926, 1.1661887586553132, 1.2299311381779416, -1.4146929897142397, -0.42980549225554004,
#         -1.4282801579740614, 0.26172301287347266, -0.5109318114918897, -0.6399495909195524, -0.733476856285442,
#         1.219652074726591, 0.08194907995352405, 0.4420398361785991, -1.184769973221183, 1.5126082924326332,
#         0.4442281271081217, -0.005079477284341147, 1.764084274265486, 0.2815940264026848, 0.2898827213634057,
#         -0.3686662696397026, 1.9125365942683656, 2.1801452989500274, -2.3915065327980467, 0.5794919897154226,
#         -1.777680085517591, 2.9015718628823604, -2.0516886588315777, 0.4146899057365943, -0.29917763685660903,
#         -0.5839240983516372, 2.1592457102697007, -0.8747902386178202, -0.5152943072876817, 0.12620001057735733,
#         1.3144109838803493, -0.5027032013330108, 1.2160353388774487, 0.7543834001473375, -3.512095548974531,
#         -0.9304382646186183, -0.30102930208709433, 0.9332135959962723, -0.52926196689098, 0.23509772959302958])

# # mean and std of (vertice-template) for each subject
# claire_mean = -0.0018
# claire_std = 0.0108
# mark_mean = -0.0007
# mark_std = 0.0094
# james_mean = -0.0019
# james_std = 0.0123

# not used for now
claire_mean = 0
claire_std = 1
mark_mean = 0
mark_std = 1
james_mean = 0
james_std = 1

def main():
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME

    # set up the device
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set up the logger
    dataset = 'a2f' # TODO
    logger = create_logger(cfg, phase="demo")

    subjects_list = ['claire', 'james', 'mark']

    # set up the model architecture
    cfg.DATASET.NFEATS = 72035
    model = get_model(cfg, dataset)

    if cfg.DEMO.EXAMPLE:
        # load audio input 
        logger.info("Loading audio from {}".format(cfg.DEMO.EXAMPLE))
        from alm.utils.demo_utils import load_example_input
        audio_path = cfg.DEMO.EXAMPLE
        assert os.path.exists(audio_path), 'audio does not exist'
        audio = load_example_input(audio_path)
    else:
        raise NotImplemented

    # load model weights
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]
    
    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # load the template
    templates = []
    for subject in subjects_list:
        template_file = os.path.join('data/A2F', f'{subject}/templates.pkl')
        logger.info("Loading template mesh from {}".format(template_file))
        with open(template_file, 'rb') as fin:
            template = pickle.load(fin,encoding='latin1')
            template = torch.Tensor(template[subject].reshape(-1))

            # template is "subtracted" from the vertice, so the mean should be "added"
            if subject == 'claire':
                template = (template + claire_mean) / claire_std
            elif subject == 'mark':
                template = (template + mark_mean) / mark_std
            elif subject == 'james':
                template = (template + james_mean) / james_std
            templates.append(template)

    # paraterize the speaking style
    speaker_to_id = {
        'claire': 0,
        'james': 1,
        'mark': 2,
    }


    for subject in subjects_list:
        speaker_id = speaker_to_id[subject]
        id = torch.zeros([1, cfg.id_dim])
        id[0, speaker_id] = 1

        template = templates[speaker_id]

        # make prediction
        logger.info("Making predictions")
        data_input = {
            'audio': audio.to(device),
            'template': template.to(device),
            'id': id.to(device),
        }
        with torch.no_grad():
            prediction = model.predict(data_input)
            vertices = prediction['vertice_pred'].squeeze().cpu().numpy()

        if subject == 'claire':
            vertices = vertices * claire_std
        elif subject == 'mark':
            vertices = vertices * mark_std
        elif subject == 'james':
            vertices = vertices * james_std
        
        print(vertices.shape, vertices.mean(), vertices.std(), vertices.max(), vertices.min())

        '''
        from plyfile import PlyData, PlyElement
        print(vertices.shape)
        print(vertices[0].shape)
        
        # Create structured array with named fields for vertex data
        vertex_data = np.zeros(len(vertices[0, :72006])//3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_data['x'] = vertices[0, :72006].reshape(-1, 3)[:, 0]
        vertex_data['y'] = vertices[0, :72006].reshape(-1, 3)[:, 1]
        vertex_data['z'] = vertices[0, :72006].reshape(-1, 3)[:, 2]
        
        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el]).write('vertices.ply')
        print(template.shape)
        vertex_data_temp = np.zeros(len(template[:72006])//3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_data_temp['x'] = template[:72006].reshape(-1, 3)[:, 0]
        vertex_data_temp['y'] = template[:72006].reshape(-1, 3)[:, 1]
        vertex_data_temp['z'] = template[:72006].reshape(-1, 3)[:, 2]
        
        el = PlyElement.describe(vertex_data_temp, 'vertex')
        PlyData([el]).write('template_vertices.ply')
        import pdb; pdb.set_trace()
        '''

        ## for testing
        #vertices = template.unsqueeze(0).repeat(vertices.shape[0],1).cpu().numpy()




        # this function is copy from faceformer
        wav_path = cfg.DEMO.EXAMPLE
        test_name = os.path.basename(wav_path).split(".")[0]
        
        output_dir = os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME)
        file_name = os.path.join(output_dir,test_name + "_" + subject + '.mp4')

        animate(vertices, wav_path, file_name, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)

if __name__ == "__main__":

    
    main()