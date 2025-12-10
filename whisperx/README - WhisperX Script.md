# 🎙️ WhisperX - Script Completo de Transcripción

Script Python completo para transcripción de audio con WhisperX, similar a [Replicate's WhisperX](https://replicate.com/victor-upmeet/whisperx).

## ✨ Características

- ✅ **Transcripción rápida** con aceleración CUDA (70x tiempo real)
- ✅ **Timestamps precisos** a nivel de palabra con alineación forzada
- ✅ **Diarización de hablantes** (identificación de quién habla)
- ✅ **Múltiples formatos de salida**: JSON, TXT, SRT, VTT
- ✅ **Detección automática de idioma** o especificación manual
- ✅ **Traducción a inglés** opcional
- ✅ **Gestión eficiente de memoria** GPU

## 📋 Requisitos

```bash
# Asegúrate de tener WhisperX instalado
source ~/whisperx_env/bin/activate

# Si no lo tienes, sigue la guía de instalación:
# https://github.com/edgardozavala/ai-tools-ubuntu-setup/blob/main/whisperx/INSTALACION_WHISPERX.md
```

## 🚀 Uso Rápido

### Transcripción básica (rápida, sin alineación)

```bash
python whisperx_script.py audio.mp3
```

**Resultado:**
- `output/audio.json` - Transcripción completa con timestamps
- `output/audio.txt` - Solo texto
- `output/audio.srt` - Subtítulos formato SRT
- `output/audio.vtt` - Subtítulos formato WebVTT

### Con alineación precisa de timestamps

```bash
# Alineación = timestamps a nivel de palabra (más lento pero más preciso)
python whisperx_script.py audio.mp3 --align
```

### Con idioma específico (más rápido)

```bash
python whisperx_script.py audio.mp3 --language es
```

### Con diarización de hablantes

```bash
# Requiere token de Hugging Face
# Obtén uno en: https://huggingface.co/settings/tokens
# Acepta el modelo: https://huggingface.co/pyannote/speaker-diarization-3.1

python whisperx_script.py audio.mp3 --diarize --hf-token YOUR_HF_TOKEN
```

**Salida con hablantes:**
```
[SPEAKER_00] Buenos días, ¿cómo estás?
[SPEAKER_01] Muy bien, gracias por preguntar.
[SPEAKER_00] Me alegro mucho de escucharlo.
```

## 📖 Ejemplos Avanzados

### 1. Podcast o entrevista con múltiples hablantes

```bash
python whisperx_script.py podcast.mp3 \
  --language es \
  --align \
  --diarize \
  --hf-token YOUR_TOKEN \
  --min-speakers 2 \
  --max-speakers 4
```

### 2. Audio con mucho ruido o silencios (ajustar VAD)

```bash
# VAD más sensible (detecta voz más fácilmente)
python whisperx_script.py audio_con_ruido.mp3 \
  --vad-onset 0.3 \
  --vad-offset 0.2

# VAD menos sensible (ignora ruido de fondo)
python whisperx_script.py audio_limpio.mp3 \
  --vad-onset 0.7 \
  --vad-offset 0.5
```

### 3. Video de YouTube (usando audio extraído)

```bash
# Primero extrae el audio con ffmpeg o yt-dlp
yt-dlp -x --audio-format mp3 "URL_DEL_VIDEO" -o video.mp3

# Luego transcribe
python whisperx_script.py video.mp3 --language es
```

### 3. Traducir audio al inglés

```bash
python whisperx_script.py audio_espanol.mp3 --task translate
```

### 4. Modelo más rápido (para pruebas)

```bash
# Modelo "base" es ~5x más rápido pero menos preciso
python whisperx_script.py audio.mp3 --model base
```

### 5. Solo JSON (para procesamiento posterior)

```bash
python whisperx_script.py audio.mp3 --output-format json
```

### 6. Procesamiento en CPU (sin GPU)

```bash
python whisperx_script.py audio.mp3 --device cpu --compute-type float32
```

## ⚙️ Parámetros Completos

```
Uso: python whisperx_script.py [OPCIONES] AUDIO_FILE

Argumentos posicionales:
  audio                 Archivo de audio a transcribir

Opciones generales:
  -o, --output-dir      Directorio de salida (default: ./output)
  -m, --model           Modelo: tiny, base, small, medium, large-v2, large-v3
  -l, --language        Código de idioma: es, en, fr, de, it, pt, ja, zh, etc.
  -t, --task            transcribe o translate (a inglés)
  -b, --batch-size      Tamaño de lote (default: 16, reducir si hay poco VRAM)
  
Opciones de hardware:
  --device              cuda o cpu
  --compute-type        float16, int8, float32
  
Opciones de VAD (Voice Activity Detection):
  --vad-onset           Umbral de inicio de voz (0.0-1.0, default: 0.500)
                        • Valores bajos (0.2-0.4): Más sensible, detecta más voz
                        • Valores altos (0.6-0.8): Menos sensible, ignora ruido
  --vad-offset          Umbral de fin de voz (0.0-1.0, default: 0.363)
                        • Valores bajos: Mantiene audio más tiempo
                        • Valores altos: Corta silencios más rápido
  
Opciones de alineación:
  --align               Activar alineación de timestamps a nivel de palabra
                        (más lento pero timestamps más precisos)
  
Opciones de diarización:
  --diarize             Activar identificación de hablantes
  --hf-token            Token de Hugging Face (requerido para diarización)
  --min-speakers        Número mínimo de hablantes
  --max-speakers        Número máximo de hablantes
  
Opciones de salida:
  --output-format       all, json, txt, srt, vtt
  -q, --quiet           Modo silencioso
```

## 📊 Modelos Disponibles

| Modelo | Parámetros | VRAM | Velocidad | Precisión | Uso Recomendado |
|--------|-----------|------|-----------|-----------|-----------------|
| `tiny` | 39M | ~1GB | 32x | ⭐⭐ | Pruebas rápidas |
| `base` | 74M | ~1GB | 16x | ⭐⭐⭐ | Transcripción rápida |
| `small` | 244M | ~2GB | 6x | ⭐⭐⭐⭐ | Balance calidad/velocidad |
| `medium` | 769M | ~5GB | 2x | ⭐⭐⭐⭐ | Alta calidad |
| `large-v2` | 1550M | ~10GB | 1x | ⭐⭐⭐⭐⭐ | Máxima precisión |
| `large-v3` | 1550M | ~10GB | 1x | ⭐⭐⭐⭐⭐ | Última versión (recomendado) |

## 🌍 Idiomas Soportados

**Con alineación completa:**
- `es` - Español
- `en` - Inglés
- `fr` - Francés
- `de` - Alemán
- `it` - Italiano
- `pt` - Portugués
- `ja` - Japonés
- `zh` - Chino
- `nl` - Holandés
- [Y más...](https://github.com/m-bain/whisperX#supported-languages)

**Otros idiomas:** Funcionan pero sin alineación de palabras (timestamps menos precisos)

## 📁 Formatos de Salida

### JSON (completo)
```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": " Hola, ¿cómo estás?",
      "words": [
        {"word": "Hola", "start": 0.5, "end": 0.9},
        {"word": "cómo", "start": 1.2, "end": 1.5},
        {"word": "estás", "start": 1.6, "end": 2.0}
      ],
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "es"
}
```

### TXT (simple)
```
Hola, ¿cómo estás?
Muy bien, gracias.
```

### SRT (subtítulos)
```
1
00:00:00,500 --> 00:00:03,200
Hola, ¿cómo estás?

2
00:00:03,500 --> 00:00:06,800
Muy bien, gracias.
```

### VTT (WebVTT)
```
WEBVTT

00:00:00.500 --> 00:00:03.200
Hola, ¿cómo estás?

00:00:03.500 --> 00:00:06.800
Muy bien, gracias.
```

## 🔧 Uso como Biblioteca Python

También puedes importar y usar la función directamente:

```python
from whisperx_script import transcribe_audio

result = transcribe_audio(
    audio_path="audio.mp3",
    output_dir="./output",
    model_name="large-v3",
    language="es",
    enable_diarization=True,
    hf_token="YOUR_HF_TOKEN"
)

# Acceder a los segmentos
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s] {segment['text']}")
```

## 🐛 Solución de Problemas

### Audio con mucho ruido de fondo
```bash
# Ajusta los umbrales VAD para ser menos sensible
python whisperx_script.py audio_ruidoso.mp3 \
  --vad-onset 0.7 \
  --vad-offset 0.5
```

### Audio muy silencioso o con pausas largas
```bash
# Ajusta los umbrales VAD para ser más sensible
python whisperx_script.py audio_silencioso.mp3 \
  --vad-onset 0.3 \
  --vad-offset 0.2
```

### Timestamps imprecisos
```bash
# Activa la alineación (más lento pero más preciso)
python whisperx_script.py audio.mp3 --align --language es
```

### Error: CUDA out of memory
```bash
# Reduce el batch_size
python whisperx_script.py audio.mp3 --batch-size 8

# O usa un modelo más pequeño
python whisperx_script.py audio.mp3 --model small
```

### Error: HF token inválido
```bash
# 1. Crea token en: https://huggingface.co/settings/tokens
# 2. Acepta el modelo: https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. Usa el token:
python whisperx_script.py audio.mp3 --diarize --hf-token hf_xxxxxxxxxxxxx
```

### Error: Alineación no disponible
```bash
# Algunos idiomas no tienen modelos de alineación
# La alineación está desactivada por defecto para mayor velocidad
# Solo actívala si necesitas timestamps muy precisos
python whisperx_script.py audio.mp3  # Sin --align es más rápido
```

### Audio muy largo (>2 horas)
```bash
# Reduce batch_size y usa compute_type int8
python whisperx_script.py audio_largo.mp3 \
  --batch-size 4 \
  --compute-type int8
```

## 📈 Rendimiento

En una **RTX 3060 (12GB VRAM)**:

### Sin alineación (modo rápido, default):
- **Audio de 10 minutos**: ~20 segundos (sin diarización), ~45 segundos (con diarización)
- **Audio de 1 hora**: ~2-3 minutos (sin diarización), ~6-8 minutos (con diarización)

### Con alineación (--align):
- **Audio de 10 minutos**: ~30 segundos (sin diarización), ~60 segundos (con diarización)
- **Audio de 1 hora**: ~3-5 minutos (sin diarización), ~8-10 minutos (con diarización)

**Modelo large-v3**: ~70x tiempo real con batch_size=16

## 🎛️ Guía de Parámetros VAD

Los parámetros `vad_onset` y `vad_offset` controlan la detección de actividad de voz:

### vad_onset (inicio de voz)
- **0.3-0.4**: Audio con mucho ruido, susurros, o voces lejanas
- **0.5** (default): Balance estándar
- **0.6-0.7**: Audio limpio en estudio, reducir falsas detecciones

### vad_offset (fin de voz)
- **0.2-0.3**: Mantener pausas naturales, respiraciones
- **0.363** (default): Balance estándar
- **0.5-0.6**: Cortar silencios largos más agresivamente

### Ejemplos prácticos:
```bash
# Podcast en estudio (limpio)
--vad-onset 0.6 --vad-offset 0.5

# Entrevista con ruido de fondo
--vad-onset 0.4 --vad-offset 0.3

# Grabación telefónica (baja calidad)
--vad-onset 0.3 --vad-offset 0.2

# Presentación con pausas largas
--vad-onset 0.5 --vad-offset 0.5
```

## 🤝 Contribuir

Este script es parte del repositorio [ai-tools-ubuntu-setup](https://github.com/edgardozavala/ai-tools-ubuntu-setup).

Mejoras bienvenidas:
- [ ] Soporte para múltiples archivos en batch
- [ ] Interfaz web con Gradio/Streamlit
- [ ] Configuración de filtros de ruido
- [ ] Export a más formatos (CSV, Excel)
- [ ] Progress bar para archivos largos

## 📄 Licencia

MIT License - Usa libremente

## 🙏 Créditos

- [WhisperX](https://github.com/m-bain/whisperX) por Max Bain
- [OpenAI Whisper](https://github.com/openai/whisper) por OpenAI
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) para diarización

---

**⭐ Si te sirvió este script, dale una estrella al repo!**
