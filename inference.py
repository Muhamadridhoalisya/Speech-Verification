import os
import torch
import numpy as np
import librosa
from torch import nn
import torch.nn.functional as F

# Fungsi untuk ekstraksi MFCC
def extract_mfcc_and_pitch(audio_path, sr=16000, n_mfcc=40):
    """
    Ekstrak fitur MFCC dan pitch dari file audio
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # Ekstrak MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Normalisasi MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    
    # Ekstrak pitch menggunakan metode YIN
    pitch = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    pitch = np.nan_to_num(pitch, nan=np.nanmean(pitch))  # Handle NaN values
    
    # Normalisasi pitch
    pitch = (pitch - np.mean(pitch)) / np.std(pitch)
    
    # Ubah pitch menjadi 2D array untuk konsistensi
    pitch = pitch.reshape(1, -1)
    
    # Gabungkan MFCC dan pitch
    combined_features = np.vstack([mfcc, pitch])
    
    return combined_features

# X-Vector Architecture
class XVectorNet(nn.Module):
    def __init__(self, input_dim=41, dropout_rate=0.45):  # Tambah 1 dimensi untuk pitch
        super(XVectorNet, self).__init__()
        
        # Frame-level features
        self.layer1 = nn.Conv1d(input_dim, 512, 5, padding=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Conv1d(512, 512, 3, padding=1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Conv1d(512, 512, 3, padding=1)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Conv1d(512, 512, 1)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.layer5 = nn.Conv1d(512, 1500, 1)
        
        # Statistics pooling
        self.stats_pooling = StatsPooling()
        
        # Segment-level features
        self.layer6 = nn.Linear(3000, 512)
        self.dropout6 = nn.Dropout(dropout_rate)
        self.layer7 = nn.Linear(512, 512)
        self.dropout7 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(512, 2)  # Binary classification
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = F.relu(self.layer4(x))
        x = self.dropout4(x)
        x = F.relu(self.layer5(x))
        
        x = self.stats_pooling(x)
        
        x = F.relu(self.layer6(x))
        x = self.dropout6(x)
        x = F.relu(self.layer7(x))
        x = self.dropout7(x)
        x = self.output(x)
        
        return x

class StatsPooling(nn.Module):
    def forward(self, x):
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat((mean, std), dim=1)

# Fungsi untuk memuat model
def load_model(model_path, input_dim=41, dropout_rate=0.45):
    model = XVectorNet(input_dim=input_dim, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Fungsi untuk melakukan inference
def inference(model, audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Ekstrak fitur dari file audio
    features = extract_mfcc_and_pitch(audio_path)
    
    # Konversi ke tensor dan tambahkan dimensi batch
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Lakukan inference
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities[:, 1].item()

# Main execution untuk inference
def main_inference(model_path, audio_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Muat model
    model = load_model(model_path).to(device)
    
    # Dapatkan semua file .wav dalam folder
    wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    
    # Lakukan inference untuk setiap file
    for wav_file in wav_files:
        audio_path = os.path.join(audio_folder, wav_file)
        predicted_class, probability = inference(model, audio_path, device)
        print(f"File: {wav_file}, Predicted Class: {predicted_class}, Probability: {probability:.4f}")

if __name__ == "__main__":
    # Path ke model yang telah disimpan
    model_path = 'output/best_overall_model.pth'
    
    # Path ke folder yang berisi file .wav untuk inference
    audio_folder = '/path/to/folder/test'
    
    # Jalankan inference
    main_inference(model_path, audio_folder)