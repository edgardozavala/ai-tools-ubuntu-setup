#!/usr/bin/env python3
"""
WhisperX - Script Completo de Transcripción
Incluye: Transcripción, Alineación, Diarización y múltiples formatos de salida
Similar a: https://replicate.com/victor-upmeet/whisperx
"""

import whisperx
import gc
import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Literal
import torch

def transcribe_audio(
    audio_path: str,
    output_dir: str = "./output",
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: Optional[str] = None,
    task: Literal["transcribe", "translate"] = "transcribe",
    # Alineación
    align_output: bool = True,
    # Diarización
    enable_diarization: bool = False,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    # Opciones de salida
    output_format: str = "all",  # all, json, txt, srt, vtt
    print_progress: bool = True
):
    """
    Transcribe audio con WhisperX incluyendo alineación y diarización opcional.
    
    Args:
        audio_path: Ruta al archivo de audio
        output_dir: Directorio para guardar resultados
        model_name: Modelo Whisper (tiny, base, small, medium, large-v2, large-v3)
        device: "cuda" o "cpu"
        compute_type: "float16", "int8" o "float32"
        batch_size: Tamaño de lote para procesamiento
        language: Código de idioma (es, en, fr, etc.) o None para detección automática
        task: "transcribe" o "translate" (traducir a inglés)
        align_output: Activar alineación para timestamps precisos
        enable_diarization: Activar identificación de hablantes
        hf_token: Token de Hugging Face para diarización
        min_speakers: Número mínimo de hablantes
        max_speakers: Número máximo de hablantes
        output_format: "all", "json", "txt", "srt", "vtt"
        print_progress: Mostrar progreso
    """
    
    # Verificar que el archivo existe
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {audio_path}")
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if print_progress:
        print(f"🎙️  WhisperX Transcription")
        print(f"📁 Audio: {audio_file.name}")
        print(f"🖥️  Device: {device}")
        print(f"🧠 Model: {model_name}")
        print(f"🌍 Language: {language or 'Auto-detect'}")
        print("-" * 50)
    
    # 1. Cargar modelo y transcribir
    if print_progress:
        print("\n[1/4] 🔄 Cargando modelo y transcribiendo...")
    
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language
    )
    
    audio = whisperx.load_audio(audio_path)
    
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        task=task
    )
    
    detected_language = result.get("language", language or "unknown")
    
    if print_progress:
        print(f"✅ Transcripción completada")
        print(f"   Idioma detectado: {detected_language}")
        print(f"   Segmentos: {len(result['segments'])}")
    
    # Liberar memoria del modelo de transcripción
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Alineación (timestamps precisos a nivel de palabra)
    aligned_result = result
    if align_output and detected_language != "unknown":
        if print_progress:
            print("\n[2/4] 🎯 Alineando timestamps a nivel de palabra...")
        
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device
            )
            
            aligned_result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            
            if print_progress:
                print("✅ Alineación completada")
            
            # Liberar memoria del modelo de alineación
            del model_a
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            if print_progress:
                print(f"⚠️  Alineación no disponible para {detected_language}: {e}")
            aligned_result = result
    
    # 3. Diarización (identificación de hablantes)
    if enable_diarization:
        if print_progress:
            print("\n[3/4] 👥 Realizando diarización de hablantes...")
        
        if not hf_token:
            print("⚠️  Token de Hugging Face requerido para diarización")
            print("   Obtén uno en: https://huggingface.co/settings/tokens")
            print("   Acepta pyannote/speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                
                aligned_result = whisperx.assign_word_speakers(
                    diarize_segments,
                    aligned_result
                )
                
                if print_progress:
                    speakers = set()
                    for segment in aligned_result["segments"]:
                        if "speaker" in segment:
                            speakers.add(segment["speaker"])
                    print(f"✅ Diarización completada")
                    print(f"   Hablantes detectados: {len(speakers)}")
                
                # Liberar memoria del modelo de diarización
                del diarize_model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                if print_progress:
                    print(f"⚠️  Error en diarización: {e}")
    
    # 4. Guardar resultados
    if print_progress:
        print(f"\n[4/4] 💾 Guardando resultados en {output_dir}...")
    
    base_name = audio_file.stem
    
    # Guardar JSON
    if output_format in ["all", "json"]:
        json_path = output_path / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aligned_result, f, ensure_ascii=False, indent=2)
        if print_progress:
            print(f"✅ JSON: {json_path}")
    
    # Guardar TXT
    if output_format in ["all", "txt"]:
        txt_path = output_path / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for segment in aligned_result["segments"]:
                speaker = f"[{segment.get('speaker', 'UNKNOWN')}] " if enable_diarization else ""
                f.write(f"{speaker}{segment['text'].strip()}\n")
        if print_progress:
            print(f"✅ TXT: {txt_path}")
    
    # Guardar SRT (subtítulos)
    if output_format in ["all", "srt"]:
        srt_path = output_path / f"{base_name}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(aligned_result["segments"], 1):
                start = format_timestamp(segment["start"], srt=True)
                end = format_timestamp(segment["end"], srt=True)
                speaker = f"[{segment.get('speaker', 'UNKNOWN')}] " if enable_diarization else ""
                text = f"{speaker}{segment['text'].strip()}"
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        if print_progress:
            print(f"✅ SRT: {srt_path}")
    
    # Guardar VTT (WebVTT)
    if output_format in ["all", "vtt"]:
        vtt_path = output_path / f"{base_name}.vtt"
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in aligned_result["segments"]:
                start = format_timestamp(segment["start"], srt=False)
                end = format_timestamp(segment["end"], srt=False)
                speaker = f"[{segment.get('speaker', 'UNKNOWN')}] " if enable_diarization else ""
                text = f"{speaker}{segment['text'].strip()}"
                
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        if print_progress:
            print(f"✅ VTT: {vtt_path}")
    
    if print_progress:
        print("\n✨ ¡Proceso completado!")
    
    return aligned_result


def format_timestamp(seconds: float, srt: bool = True) -> str:
    """Formatea timestamp para SRT o VTT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX - Transcripción de audio con alineación y diarización",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Transcripción básica
  python whisperx_script.py audio.mp3

  # Transcripción con idioma específico
  python whisperx_script.py audio.mp3 --language es

  # Con diarización (requiere token HF)
  python whisperx_script.py audio.mp3 --diarize --hf-token YOUR_TOKEN

  # Traducir a inglés
  python whisperx_script.py audio.mp3 --task translate

  # Modelo más rápido (menos preciso)
  python whisperx_script.py audio.mp3 --model base

  # Solo salida JSON
  python whisperx_script.py audio.mp3 --output-format json
        """
    )
    
    parser.add_argument("audio", help="Archivo de audio a transcribir")
    parser.add_argument("-o", "--output-dir", default="./output", 
                       help="Directorio de salida (default: ./output)")
    parser.add_argument("-m", "--model", default="large-v3",
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                       help="Modelo Whisper (default: large-v3)")
    parser.add_argument("-l", "--language", default=None,
                       help="Código de idioma (es, en, fr, etc.). Auto-detect si no se especifica")
    parser.add_argument("-t", "--task", default="transcribe",
                       choices=["transcribe", "translate"],
                       help="Transcribir o traducir a inglés (default: transcribe)")
    parser.add_argument("-b", "--batch-size", type=int, default=16,
                       help="Tamaño de lote (default: 16)")
    parser.add_argument("--device", default="cuda",
                       choices=["cuda", "cpu"],
                       help="Dispositivo (default: cuda)")
    parser.add_argument("--compute-type", default="float16",
                       choices=["float16", "int8", "float32"],
                       help="Tipo de computación (default: float16)")
    parser.add_argument("--no-align", action="store_true",
                       help="Deshabilitar alineación de timestamps")
    parser.add_argument("--diarize", action="store_true",
                       help="Activar diarización de hablantes")
    parser.add_argument("--hf-token", default=None,
                       help="Token de Hugging Face para diarización")
    parser.add_argument("--min-speakers", type=int, default=None,
                       help="Número mínimo de hablantes")
    parser.add_argument("--max-speakers", type=int, default=None,
                       help="Número máximo de hablantes")
    parser.add_argument("--output-format", default="all",
                       choices=["all", "json", "txt", "srt", "vtt"],
                       help="Formato de salida (default: all)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Modo silencioso")
    
    args = parser.parse_args()
    
    try:
        transcribe_audio(
            audio_path=args.audio,
            output_dir=args.output_dir,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            language=args.language,
            task=args.task,
            align_output=not args.no_align,
            enable_diarization=args.diarize,
            hf_token=args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            output_format=args.output_format,
            print_progress=not args.quiet
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
