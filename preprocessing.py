import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

def trim_wav(input_file, output_file, target_duration_minutes=24, target_duration_seconds=10):
    """
    Memotong file WAV sesuai durasi target
    """
    try:
        # Load file WAV
        audio = AudioSegment.from_wav(input_file)
        
        # Konversi target durasi ke milidetik
        target_duration_ms = (target_duration_minutes * 60 + target_duration_seconds) * 1000
        
        # Cek apakah durasi audio melebihi target
        if len(audio) <= target_duration_ms:
            print(f"File {input_file} memiliki durasi kurang dari target, diloncati")
            return None
            
        # Potong audio sesuai durasi yang diinginkan
        trimmed_audio = audio[:target_duration_ms]
        
        # Export file yang sudah dipotong
        trimmed_audio.export(output_file, format="wav")
        
        print(f"File berhasil dipotong menjadi {target_duration_minutes} menit {target_duration_seconds} detik")
        return output_file
        
    except Exception as e:
        print(f"Terjadi kesalahan saat memotong file: {str(e)}")
        return None

def segment_wav_file(input_file, output_dir, segment_duration=5, overlap_percentage=0.5):
    """
    Memotong file WAV menjadi segmen-segmen dengan overlap
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input WAV file
    sample_rate, audio_data = wavfile.read(input_file)
    
    # Convert segment duration to samples
    samples_per_segment = int(segment_duration * sample_rate)
    samples_overlap = int(samples_per_segment * overlap_percentage)
    
    # Calculate step size between segments
    step_size = samples_per_segment - samples_overlap
    
    # Segment the audio
    segmented_files = []
    for start in range(0, len(audio_data) - samples_per_segment + 1, step_size):
        end = start + samples_per_segment
        segment = audio_data[start:end]
        
        # Generate output filename dengan format "sound_segment_X.wav"
        segment_filename = f"ridho_segment_{start//step_size}.wav"
        output_path = os.path.join(output_dir, segment_filename)
        
        # Write segmented audio to file
        wavfile.write(output_path, sample_rate, segment)
        segmented_files.append(output_path)
    
    print(f"Berhasil membuat {len(segmented_files)} segmen")
    return segmented_files

def process_audio_pipeline(input_wav_file, output_dir):
    """
    Fungsi utama untuk memproses single file WAV
    """
    try:
        # Buat direktori output jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Trim to 24:10
        trimmed_wav = os.path.join(output_dir, "trimmed.wav")
        trimmed_result = trim_wav(input_wav_file, trimmed_wav)
        if not trimmed_result:
            return
            
        # Step 2: Create segments
        segments_dir = os.path.join(output_dir, "segments")
        segmented_files = segment_wav_file(
            trimmed_wav,
            segments_dir,
            segment_duration=5,
            overlap_percentage=0.5
        )
        
        print("Proses selesai!")
        return segmented_files
        
    except Exception as e:
        print(f"Terjadi kesalahan dalam pipeline: {str(e)}")
        return None

# Contoh penggunaan
if __name__ == "__main__":
    # Sesuaikan path file input dan direktori output
    input_wav = "full recording.wav"
    output_dir = "/nama/path/to/target/dataset"
    
    process_audio_pipeline(input_wav, output_dir)