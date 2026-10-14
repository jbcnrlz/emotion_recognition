import torch,os,sys,argparse
from diffusers.models import AutoencoderKL
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet

def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[0])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()))) + ',%s\n' % (f[1]))

def main():
    parser = argparse.ArgumentParser(description='Extract latent features with AutoencoderKL')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--outputCSVLatent', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,termsQuantity=151)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)
    outputFile = []
    with torch.no_grad():
        for img,label,pathfile in val_loader:
            img = img.to(device)
            encodedFeature = vae.encode(img).latent_dist.mode().view((img.shape[0],-1)).cpu()
            outputFile += list(zip(encodedFeature,pathfile))
        
    generateCSVFile(args.outputCSVLatent,outputFile)

if __name__ == '__main__':
    main()