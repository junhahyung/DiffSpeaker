import os
import pickle
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate

import numpy as np

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
    dataset = 'BIWI' # TODO
    logger = create_logger(cfg, phase="demo")

    # set up the model architecture
    cfg.DATASET.NFEATS = 70110
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

    #import pdb; pdb.set_trace()
    #state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    #model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # load the template
    logger.info("Loading template mesh from {}".format(cfg.DEMO.TEMPLATE))
    template_file = cfg.DEMO.TEMPLATE
    with open(template_file, 'rb') as fin:
        template = pickle.load(fin,encoding='latin1')
        subject_id = cfg.DEMO.ID
        assert subject_id in template, f'{subject_id} is not a subject included'
        template = torch.Tensor(template[subject_id].reshape(-1))

    # paraterize the speaking style
    speaker_to_id =  {
        "F2": 0,
        "F3": 1,
        "F4": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
    }
    if cfg.DEMO.ID in speaker_to_id:
        speaker_id = speaker_to_id[cfg.DEMO.ID]
        id = torch.zeros([1, cfg.id_dim])
        id[0, speaker_id] = 1
    else:
        id = torch.zeros([1, cfg.id_dim])
        id[0, 0] = 1
    
    print(id)

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

    


    from plyfile import PlyData, PlyElement
    '''
    print(vertices.shape)
    print(vertices[0].shape)
    
    # Create structured array with named fields for vertex data
    vertex_data = np.zeros(len(vertices[0])//3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data['x'] = vertices[0].reshape(-1, 3)[:, 0]
    vertex_data['y'] = vertices[0].reshape(-1, 3)[:, 1]
    vertex_data['z'] = vertices[0].reshape(-1, 3)[:, 2]
    
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el]).write('vertices.ply')
    import pdb; pdb.set_trace()
    '''

    '''
    vertex_data_temp = np.zeros(len(template)//3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data_temp['x'] = template.reshape(-1, 3)[:, 0]
    vertex_data_temp['y'] = template.reshape(-1, 3)[:, 1]
    vertex_data_temp['z'] = template.reshape(-1, 3)[:, 2]
    
    el = PlyElement.describe(vertex_data_temp, 'vertex')
    PlyData([el]).write('template_vertices.ply')
    import pdb; pdb.set_trace()
    '''



    # this function is copy from faceformer
    wav_path = cfg.DEMO.EXAMPLE
    test_name = os.path.basename(wav_path).split(".")[0]
    
    output_dir = os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME)
    file_name = os.path.join(output_dir,test_name + "_" + subject_id + '.mp4')

    # for testing
    #########################################################
    #vertices = template.unsqueeze(0).repeat(vertices.shape[0],1).cpu().numpy()
    #########################################################
    

    animate(vertices, wav_path, file_name, cfg.DEMO.PLY, fps=25, use_tqdm=True, multi_process=True)

if __name__ == "__main__":
    main()