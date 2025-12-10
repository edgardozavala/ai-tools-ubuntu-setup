# Guía Completa: Instalación WhisperX desde Cero
## Pop!_OS 22.04 LTS + NVIDIA RTX 3060

**Esta guía garantiza una instalación funcional de WhisperX con CUDA, VAD y diarización.**

---

## 📋 Requisitos del Sistema

- **OS**: Pop!_OS 22.04 LTS (Ubuntu 22.04 compatible)
- **GPU**: NVIDIA GeForce RTX 3060 (Compute Capability 8.6)
- **Python**: 3.10.12

---

## 🚀 Paso 1: Instalar Driver NVIDIA

```bash
# Actualiza el sistema
sudo apt update && sudo apt upgrade -y

# Instala el driver NVIDIA 565 (o superior)
sudo apt install system76-driver-nvidia -y

# Alternativa manual si lo anterior no funciona:
# sudo apt install nvidia-driver-565 -y

# Reinicia el sistema
sudo reboot

# Después del reinicio, verifica el driver
nvidia-smi
# Debe mostrar: Driver Version: 565.77 o superior
```

---

## 🔧 Paso 2: Instalar CUDA 12.x

```bash
# Descarga e instala CUDA Toolkit 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8 -y

# Configura las variables de entorno
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verifica CUDA (puede mostrar 11.5, es normal - nvcc != runtime)
nvcc --version
```

**NOTA**: Es normal que `nvcc --version` muestre CUDA 11.5. El runtime de CUDA 12.8 se usa correctamente.

---

## 📦 Paso 3: Instalar cuDNN 8 y cuDNN 9 (CRÍTICO)

WhisperX necesita **AMBAS versiones** coexistiendo:
- cuDNN 8: Para Pyannote VAD (modelos antiguos)
- cuDNN 9: Para PyTorch 2.8.0 (modelos modernos)

```bash
# Instala cuDNN 8 (para Pyannote)
sudo apt install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2

# Instala cuDNN 9 (para Torch 2.8.0)
wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn9-cuda-12=9.17.0.29-1

# CRÍTICO: Crea symlinks para compatibilidad de versiones
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.17.0 /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.17.0 /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.1.0
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.17.0 /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.17.0 /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.1.0

# Actualiza el cache de librerías
sudo ldconfig

# Verifica que ambas versiones estén disponibles
ldconfig -p | grep libcudnn_ops_infer.so.8  # Para Pyannote
ldconfig -p | grep libcudnn_ops.so.9         # Para Torch
```

---

## 🐍 Paso 4: Crear Entorno Virtual e Instalar WhisperX

```bash
# Instala dependencias del sistema
sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg git

# Crea el entorno virtual
cd ~
python3.10 -m venv whisperx_env
source whisperx_env/bin/activate

# Actualiza pip
pip install --upgrade pip setuptools wheel

# Instala PyTorch 2.8.0 con CUDA 12.8
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Instala WhisperX (instalará todas las dependencias automáticamente)
pip install git+https://github.com/m-bain/whisperx.git

# Verifica la instalación
python << EOF
import torch
import whisperx
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Torch: {torch.__version__}")
print(f"✅ WhisperX: {whisperx.__version__}")
EOF
```

**Resultado esperado:**
```
✅ CUDA disponible: True
✅ GPU: NVIDIA GeForce RTX 3060
✅ Torch: 2.8.0
✅ WhisperX: 3.5.0 o 3.7.4
```

---

## 🧪 Paso 5: Prueba Funcional

Crea un script de prueba:

```bash
cat > test_whisperx.py << 'EOF'
import whisperx
import torch

print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

# Carga modelo con VAD
model = whisperx.load_model("base", device="cuda", compute_type="float16")
print("✅ Modelo cargado exitosamente con VAD")

# Descarga audio de prueba (opcional)
# wget https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav -O test.wav

# Si tienes un audio, prueba la transcripción:
# audio = whisperx.load_audio("test.wav")
# result = model.transcribe(audio, language="es")
# print(result["segments"])

print("🎉 ¡WhisperX instalado correctamente!")
EOF

python test_whisperx.py
```

---

## 📝 Configuración Final que Funciona

```
Sistema Operativo: Pop!_OS 22.04 LTS
Kernel:            6.12.10-76061203-generic
Driver NVIDIA:     565.77
CUDA Runtime:      12.8
cuDNN:             8.9.7 (Pyannote) + 9.17.0 (Torch)
Python:            3.10.12

Paquetes Python:
├── torch==2.8.0
├── torchaudio==2.8.0
├── whisperx==3.5.0 o 3.7.4
├── ctranslate2==4.4.0
├── faster-whisper==1.2.1
├── pyannote.audio==3.4.0
├── transformers==4.57.3
└── numpy==2.2.6
```

---

## 🎯 Script de Uso Básico

```python
import whisperx

# Carga modelo (usa "large-v3" para mejor precisión)
model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")

# Transcribe audio
audio_path = "/ruta/a/tu/audio.mp3"
result = model.transcribe(audio_path, language="es", batch_size=16)

# Muestra resultados
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

# Guarda en archivo
with open("transcripcion.txt", "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        f.write(f"{segment['text']}\n")
```

---

## 🐛 Solución de Problemas Comunes

### Error: "libcudnn_ops_infer.so.8: cannot open shared object"
```bash
# Verifica que cuDNN 8 esté instalado
ldconfig -p | grep libcudnn_ops_infer.so.8
# Si no aparece, reinstala:
sudo apt install -y libcudnn8=8.9.7.29-1+cuda12.2
sudo ldconfig
```

### Error: "Unable to load libcudnn_cnn.so.9.1.0"
```bash
# Recrea los symlinks
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.17.0 /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
sudo ldconfig
```

### Error: "UnpicklingError" con Pyannote
Esto indica que NO tienes cuDNN 8 instalado. Vuelve al Paso 3.

### CUDA no disponible en Python
```bash
# Verifica driver
nvidia-smi

# Reinstala PyTorch
pip uninstall torch torchaudio
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## ✅ Checklist de Verificación

- [ ] `nvidia-smi` muestra Driver 565.77+
- [ ] `nvcc --version` ejecuta sin error
- [ ] `ldconfig -p | grep libcudnn_ops_infer.so.8` encuentra la librería
- [ ] `ldconfig -p | grep libcudnn_ops.so.9` encuentra la librería
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` retorna True
- [ ] `python test_whisperx.py` ejecuta sin errores

---

## 🎉 ¡Listo!

Siguiendo estos pasos exactos, tendrás WhisperX funcionando con:
- ✅ Aceleración CUDA completa
- ✅ VAD (Voice Activity Detection) con Pyannote
- ✅ Soporte para diarización de hablantes
- ✅ Sin crashes ni core dumps

**Tiempo estimado de instalación**: 30-45 minutos

**Nota**: Guarda esta guía en un lugar seguro. Si reinstalaras el sistema, simplemente sigue estos pasos en orden y funcionará a la primera.
