import torch
import cv2
import sys
import os
import glob
import numpy as np
import torch.nn.functional as F
from urllib.request import urlretrieve
import tempfile

def initialize_device():
    """
    Inizializza e verifica il dispositivo CUDA una sola volta
    """
    print(f"PyTorch versione: {torch.__version__}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU disponibile: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo dispositivo: {device}")
    return device

def upscale_image_simple(image_path, scale_factor=2, device=None, face_mode=False):
    """
    Versione semplificata che usa PyTorch direttamente per super-resolution
    """
    # Verifica se il file esiste
    if not os.path.exists(image_path):
        print(f"Errore: il file {image_path} non esiste.")
        return False

    # Carica l'immagine originale
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Verifica se il file è un'immagine
    if image is None:
        print(f"Errore: il file {image_path} non è un'immagine valida.")
        return False

    # Usa il dispositivo passato come parametro o inizializza uno nuovo
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Verifica se scaricare il modello Real-ESRGAN
        if scale_factor in [2, 4]:
            model_path = download_esrgan_model(scale_factor, face_mode)
            if model_path:
                mode_name = "face" if face_mode else "general"
                print(f"  Utilizzo modello Real-ESRGAN ({mode_name}) per migliore qualità")
        
        # Converti l'immagine BGR (OpenCV) in RGB e normalizza
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Converti in tensor PyTorch
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) -> (1,C,H,W)
        image_tensor = image_tensor.to(device)

        # Applica super-resolution usando interpolazione avanzata su GPU
        with torch.no_grad():
            if scale_factor <= 2:
                # Per fattori bassi, usa bicubic
                upscaled_tensor = F.interpolate(
                    image_tensor, 
                    scale_factor=scale_factor, 
                    mode='bicubic', 
                    align_corners=False
                )
            else:
                # Per fattori alti, combina più passaggi
                current_scale = 1
                result_tensor = image_tensor
                while current_scale < scale_factor:
                    step_scale = min(2, scale_factor / current_scale)
                    result_tensor = F.interpolate(
                        result_tensor, 
                        scale_factor=step_scale, 
                        mode='bicubic', 
                        align_corners=False
                    )
                    current_scale *= step_scale
                upscaled_tensor = result_tensor

        # Converti back a numpy e OpenCV format
        upscaled_np = upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        upscaled_np = (upscaled_np * 255.0).clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)

        # Ottieni il nome del file e l'estensione
        base_name, ext = os.path.splitext(image_path)
        suffix = f"_PyTorch_x{scale_factor}"
        if face_mode:
            suffix += "_face"
        output_path = f"{base_name}{suffix}{ext}"

        # Salva l'immagine ingrandita
        cv2.imwrite(output_path, result)
        print(f"L'immagine {image_path} è stata ingrandita e salvata come {output_path}.")
        return True

    except Exception as e:
        print(f"Errore durante l'elaborazione di {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_esrgan_model(scale_factor, face_mode=False):
    """
    Scarica i modelli Real-ESRGAN (migliori modelli disponibili)
    """
    if face_mode:
        # Modelli specializzati per volti/ritratti
        models = {
            2: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "name": "RealESRGAN_x2plus_face"
            },
            4: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 
                "name": "RealESRGAN_x4plus_face"
            }
        }
        model_type = "face"
    else:
        # Modelli Real-ESRGAN standard (migliori per uso generale)
        models = {
            2: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "name": "RealESRGAN_x2plus"
            },
            4: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "name": "RealESRGAN_x4plus"
            }
        }
        model_type = "general"
    
    if scale_factor not in models:
        print(f"Fattore di scala {scale_factor} non supportato per {model_type} mode.")
        return None
        
    model_info = models[scale_factor]
    model_path = f"./models/{model_info['name']}.pth"
    os.makedirs("./models", exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Scaricamento modello {model_info['name']} x{scale_factor}...")
        try:
            urlretrieve(model_info["url"], model_path)
            print(f"Modello scaricato: {model_path}")
        except Exception as e:
            print(f"Errore nel download: {e}")
            return None
    else:
        print(f"Modello {model_info['name']} già presente.")
    
    return model_path

def check_dependencies():
    """Verifica che tutte le dipendenze siano installate"""
    try:
        import torch
        import cv2
        import numpy as np
        print("✓ Tutte le dipendenze sono installate")
        return True
    except ImportError as e:
        print(f"✗ Dipendenza mancante: {e}")
        print("Installa le dipendenze con:")
        print("pip install torch torchvision opencv-python numpy")
        return False

if __name__ == "__main__":
    # Verifica dipendenze
    if not check_dependencies():
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Uso: python SuperResizePyTorch.py <image_path_or_pattern> [--x <scale_factor>] [--face]")
        print("Fattori di scala supportati: 2, 3, 4, 8")
        print("Modelli:")
        print("  Default: Real-ESRGAN x2plus/x4plus (migliore qualità generale)")
        print("  --face: Modelli ottimizzati per volti e ritratti")
        print("Esempi:")
        print("  python SuperResizePyTorch.py immagine.jpg")
        print("  python SuperResizePyTorch.py *.jpg --x 4")
        print("  python SuperResizePyTorch.py ritratto.png --x 2 --face")
        print("  python SuperResizePyTorch.py ./photos/*.jpg --x 4 --face")
    else:
        # Gestione dei parametri
        scale_factor = 2  # Valore di default
        face_mode = False
        
        # Verifica se c'è il flag --face
        if '--face' in sys.argv:
            face_mode = True
            # Rimuovi --face dalla lista degli argomenti
            args_without_face = [arg for arg in sys.argv if arg != '--face']
        else:
            args_without_face = sys.argv[:]
        
        # Verifica se c'è il parametro --x
        if len(args_without_face) >= 4 and args_without_face[-2] == '--x':
            try:
                scale_factor = int(args_without_face[-1])
                if scale_factor not in [2, 3, 4, 8]:
                    print("Fattore di scala non supportato. Supportati: 2, 3, 4, 8.")
                    print("Utilizzo il valore di default (2).")
                    scale_factor = 2
                # Rimuovi gli ultimi due argomenti (--x e scale_factor)
                pattern = ' '.join(args_without_face[1:-2])
            except ValueError:
                print("Valore di ingrandimento non valido, utilizzo il valore di default (2).")
                pattern = ' '.join(args_without_face[1:])
        else:
            pattern = ' '.join(args_without_face[1:])
        
        # Espandi il pattern con glob per trovare tutti i file corrispondenti
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # Se glob non trova niente, prova a trattarlo come un singolo file
            if os.path.exists(pattern):
                matching_files = [pattern]
            else:
                print(f"Errore: nessun file trovato per il pattern '{pattern}'")
                sys.exit(1)
        
        print(f"Trovati {len(matching_files)} file da processare:")
        for file_path in matching_files:
            print(f"  - {file_path}")
        
        # Inizializza il dispositivo una sola volta
        device = initialize_device()
        print(f"Modalità: {'Face/Ritratti' if face_mode else 'Generale'}")
        print()
        
        processed = 0
        failed = 0
        
        for i, file_path in enumerate(matching_files, 1):
            print(f"[{i}/{len(matching_files)}] Processando: {file_path}")
            if upscale_image_simple(file_path, scale_factor, device, face_mode):
                processed += 1
            else:
                failed += 1
            print()
        
        print(f"Risultati:")
        print(f"  File processati con successo: {processed}")
        print(f"  File con errori: {failed}")
        print(f"  Totale: {len(matching_files)}")
        print(f"  Fattore di scala utilizzato: x{scale_factor}")
        print(f"  Modalità: {'Face/Ritratti' if face_mode else 'Generale'}")
        print(f"  Dispositivo utilizzato: {device}")
