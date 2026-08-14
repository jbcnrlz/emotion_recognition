import torch, torch.nn as nn, torch.optim as optim, sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.MIntPainDataset import MIntPAINProtocolDataset
from networks.EmotionResnetVA import PainIndexCrossAttention

def train_pain_model(
    features_csv, 
    train_txt, 
    val_txt, 
    batch_size=64, 
    epochs=50, 
    lr=1e-4, 
    window_size=30,
    save_dir="./checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Inicialização do TensorBoard
    log_dir = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard inicializado. Para visualizar, execute no terminal: tensorboard --logdir={log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento no dispositivo: {device}")

    # 2. Preparação dos Dados
    print("Carregando datasets...")
    train_dataset = MIntPAINProtocolDataset(features_csv, train_txt, window_size=window_size)
    val_dataset = MIntPAINProtocolDataset(features_csv, val_txt, window_size=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. Instanciação do Modelo, Loss e Otimizador
    model = PainIndexCrossAttention(window_size=window_size).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    # 4. Loop de Treinamento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader.dataset)

        # 5. Loop de Validação
        model.eval()
        val_loss = 0.0
        val_mae = 0.0 
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                predictions, _ = model(batch_x)
                
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                mae = torch.abs(predictions - batch_y).mean()
                val_mae += mae.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        scheduler.step(val_loss)

        # 6. Logs no TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        
        writer.add_scalar('Loss/Train (MSE)', train_loss, epoch)
        writer.add_scalar('Loss/Validation (MSE)', val_loss, epoch)
        writer.add_scalar('Metric/Validation (MAE)', val_mae, epoch)
        writer.add_scalar('Hyperparameters/Learning_Rate', current_lr, epoch)

        # 7. Salvamento do Melhor Modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_pain_index_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"--> Novo melhor modelo salvo em {save_path}")

    writer.close()
    print("Treinamento concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento do Modelo de Índice de Dor (Cross-Attention)")
    
    # Argumentos Obrigatórios
    parser.add_argument("--features", type=str, required=True, 
                        help="Caminho para o CSV contendo as features unificadas.")
    parser.add_argument("--train_txt", type=str, required=True, 
                        help="Caminho para o arquivo TXT de protocolo de treino.")
    parser.add_argument("--val_txt", type=str, required=True, 
                        help="Caminho para o arquivo TXT de protocolo de validação.")
    
    # Hiperparâmetros (Opcionais com valores default)
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Tamanho do batch (default: 64).")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Número de épocas para treinamento (default: 50).")
    parser.add_argument("--lr", type=float, default=5e-4, 
                        help="Taxa de aprendizado inicial (default: 0.0005).")
    parser.add_argument("--window_size", type=int, default=30, 
                        help="Tamanho da janela temporal T (default: 30).")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", 
                        help="Diretório para salvar os checkpoints e logs do TensorBoard (default: ./checkpoints).")
    
    args = parser.parse_args()
    
    train_pain_model(
        features_csv=args.features,
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        window_size=args.window_size,
        save_dir=args.save_dir
    )