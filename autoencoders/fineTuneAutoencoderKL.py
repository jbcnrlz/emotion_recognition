from diffusers.models import AutoencoderKL
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    print('oi')

if __name__ == '__main__':
    main()
