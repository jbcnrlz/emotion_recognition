import torch.utils.data, os, sys, torch, bimpy, argparse, logging
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from ALAE.net import *
from ALAE.model import Model
from ALAE.defaults import get_cfg_defaults
from ALAE.checkpointer import Checkpointer
from DatasetClasses.AffectNet import AffectNet
from PIL import Image

def encode(x,model,layer_count):
    Z, _ = model.encode(x, layer_count - 1, 1)
    Z = Z.repeat(1, model.mapping_f.num_layers, 1)
    return Z

def getLatentFeatures(model,img,attribute_values,W,layer_count):    
    im = img[0]
    #im.requires_grad = True
    im.to('cuda')
    x =  im / 127.5 - 1.
    if x.shape[0] == 4:
        x = x[:3]

    needed_resolution = model.decoder.layer_to_resolution[-1]
    while x.shape[2] > needed_resolution:
        x = F.avg_pool2d(x, 2, 2)
    if x.shape[2] != needed_resolution:
        x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))
    latents_original = encode(x[None, ...].cuda(),model,layer_count)
    return latents_original.cpu()

def main():
    parser = argparse.ArgumentParser(description='Extract latent features with ALAE')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--outputCSVLatent', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.to('cuda')
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f
    dlatent_avg = model.dlatent_avg

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)
    extra_checkpoint_data = checkpointer.load()
    indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]
    layer_count = cfg.MODEL.LAYER_COUNT
    attribute_values = [bimpy.Float(0) for i in indices]    
    W = [torch.tensor(np.load("ALAE/principal_directions/direction_%d.npy" % i), dtype=torch.float32) for i in indices]
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,termsQuantity=151)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) 
    outputFile = []
    for img,label,pathfile in val_loader:
        ltOriginal = getLatentFeatures(model,img,attribute_values,W,layer_count)
        outputFile.append((ltOriginal.view((1,-1)).cpu(),pathfile[0]))
        
    generateCSVFile(args.outputCSVLatent,outputFile)

def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[1])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()[0]))) + ',%s\n' % (f[1]))

if __name__ == '__main__':
    main()