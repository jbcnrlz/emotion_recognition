import argparse, re
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from bert_score import score

def loadResultsFile(pathFile):
    returnData = {}
    with open(pathFile, 'r') as file:
        for f in file:
            contFile = f.strip()
            data = contFile.split(' - ')
            imageFileNumber = re.sub(r'\D', '', data[-1])
            returnData[imageFileNumber] = [data[0]]
    return returnData

def organizeResults(estimations, groundTruth):
    returnEstimations = {}
    returnGroundTruth = {}
    for key in estimations:
        if key in groundTruth:
            returnEstimations[key] = estimations[key]
            returnGroundTruth[key] = groundTruth[key]

    return returnEstimations, returnGroundTruth

def main():
    
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--fileEstim', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--fileGt', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    estimations = loadResultsFile(args.fileEstim)
    groundTruth = loadResultsFile(args.fileGt)
    estimations, groundTruth = organizeResults(estimations, groundTruth)

    bleu_scorer = Bleu(n=4)  # BLEU-1 até BLEU-4
    #meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    #spice_scorer = Spice()

    bleu_scores, _ = bleu_scorer.compute_score(groundTruth, estimations)
    #meteor_score, _ = meteor_scorer.compute_score(groundTruth, estimations)
    rouge_score, _ = rouge_scorer.compute_score(groundTruth, estimations)
    cider_score, _ = cider_scorer.compute_score(groundTruth, estimations)
    #spice_score, _ = spice_scorer.compute_score(groundTruth, estimations)

    print(f"BLEU: {bleu_scores}")  # [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
    #print(f"METEOR: {meteor_score}")
    print(f"ROUGE-L: {rouge_score}")
    print(f"CIDEr: {cider_score}")
    #print(f"SPICE: {spice_score}")

    avgP, avgR, avgF1 = 0, 0, 0
    P, R, F1 = score([estimations[e][0] for e in estimations], [[groundTruth[it][0]] for it in groundTruth], lang="en", verbose=True)
    avgP += P.mean().item()
    avgR += R.mean().item()
    avgF1 += F1.mean().item()

    print(f"BERTScore Precision (P): {avgP:.4f}")
    print(f"BERTScore Recall (R): {avgR:.4f}")
    print(f"BERTScore F1: {avgF1:.4f}")

if __name__ == '__main__':
    main()
