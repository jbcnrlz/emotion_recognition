#!/bin/bash
# Torne o arquivo executável com: chmod +x run_experiments.sh

echo "Treinando modelo com Knowledge Enhanced loss..."
python3 scripts/train/finetuneGMMHead.py --pathBase /mnt/17786ee3-9d2c-418f-9dea-5e545e6b46ee/Datasets/AffectNetFullMerge --batch 25 --learningRate 1e-4 --output experiments/simplenetworkNewDistros_KNOWLEDGEGUIDEDCONSISTENCY_loss_14 --epochs 20 --numberOfClasses 15 --model simpleNetwork --useDominance True --mainLossFunc KNOWLEDGEGUIDEDCONSISTENCY --visualizeConflict True --typeExperiment PROBS_VAD_EXP --pretrainedResnet pretrained/16backbone.pth --imageSize 224 

echo "Experimentos concluídos!"