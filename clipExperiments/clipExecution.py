import argparse, torch, os, sys, clip, numpy as np
from PIL import Image
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet

def saveToCSV(preds,files,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('valence,arousal,file\n')
        for idx, p in enumerate(preds):
            pcsv.write('%f,%f,%s\n' % (p[0],p[1],files[idx]))

def test():
    parser = argparse.ArgumentParser(description='Do classification with CLIP')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("Model loaded")
    print(model)
    dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment="EXP")
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)    
    #classesDist = np.array(["a person without any facial expression", "a happy person", "a sad person", "a surprised person", "a person with fear", "a disgusted person", "an angered person", "a person showing contempt"])
    classesDist = np.array(["a person with a neutral facial expression", "a person showing emotion"])
    original = np.array(["neutral", "non-neutral"])
    text = clip.tokenize(classesDist).to(device)
    acc = [0,0]
    outputString = []
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))            
            _, label, pathsForFiles = data
            finalLabel = label.cpu().numpy()[0]
            finalLabel = int(finalLabel > 0)
            image = preprocess(Image.open(pathsForFiles[0])).unsqueeze(0).to(device)    
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()            
            acc[int(np.argsort(-probs[0])[0] == finalLabel)] += 1
            outputString.append([pathsForFiles[0],' | '.join(classesDist[np.argsort(-probs[0])]),original[finalLabel],'match' if int(np.argsort(-probs[0])[0] == finalLabel) else 'not match'])
    print("Accuracy = %.2f" % (acc[1] / sum(acc)))

    with open("anotations_onlyemotions.csv",'w') as anfile:
        anfile.write('filepath,semanticannotation,truelabel,result\n')
        for o in outputString:
            anfile.write(','.join(o) + '\n')

if __name__ == '__main__':
    test()