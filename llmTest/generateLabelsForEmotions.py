import numpy as np, os, sys, argparse, pandas as pd 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")


def loadRanksFiles(datasetPath):
    returnRanksFiles = []
    images = getFilesInPath(os.path.join(datasetPath,'images'))
    pathFiles = []
    for i in images:
        fileName = i.split(os.path.sep)[-1][:-4]
        with open(os.path.join(datasetPath,'annotations',f"{fileName}_prob_rank.txt"), 'r') as file:
            for f in file:
                returnRanksFiles.append(list(map(float, f.split(','))))
        pathFiles.append(i.split(os.path.sep)[-1])

    return np.array(returnRanksFiles), pathFiles


def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    ranks = None

    if args.pathBase[-4] == '.':
        csvFile = np.array(pd.read_csv(args.pathBase))
        ranks = csvFile[:,:-1].astype(np.float64)
        files = csvFile[:,-1]
    else:
        ranks, files = loadRanksFiles(args.pathBase)
        
    hf_token = None
    login(hf_token)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    emotions = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'angry', 'contempt', 'serene', 'contemplative', 'secure', 'untroubled', 'quiet']

    outputsToFile = []

    for idx2, r in enumerate(ranks):
        if len(outputsToFile) >= 1302:
            break
        print(f"Doing file numer {idx2}")
        messageEmotions = ''
        for idx in range(len(r)):
            messageEmotions += f"{emotions[idx]}: {r[idx]:.2f}, "

        #messageEmotions[-2] = '.'

        messages = [
            {"role": "system", "content": "You are evaluating the display of different emotions based on their probabilities. Answer each input with a caption that better describes the emotions distribution"},
            {"role": "user", "content": f"Given the set of terms below describing the emotional state of a face, each term being associated with a probability, generate a caption describing the emotional state: {messageEmotions}."},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))
        outputsToFile.append([tokenizer.decode(response, skip_special_tokens=True),files[idx2]])        

        
    with open('emotions.txt', 'w') as f:
        for i in outputsToFile:
            f.write(f"{i[0]} - {i[1]}\n")

if __name__ == "__main__":
    main()