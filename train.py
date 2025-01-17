import numpy as np
import librosa
import os
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
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

def compute_eer(y_true, y_scores):
    """
    Menghitung Equal Error Rate (EER) dari predicted scores
    
    Args:
        y_true: Label yang benar (ground truth)
        y_scores: Probability scores dari model (untuk kelas positif)
    
    Returns:
        eer: Equal Error Rate
        threshold: Threshold optimal di titik EER
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Cari titik di mana FPR dan FNR berpotongan
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer, eer_threshold

def evaluate_model(model, data_loader, device):
    """
    Evaluasi model dan hitung EER
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            scores = F.softmax(output, dim=1)[:, 1]  # Probability untuk kelas positif
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    return eer, threshold

def extract_number(file_name):
    """Extract number from filename for proper sorting"""
    match = re.search(r'segment_(\d+)', file_name)
    if match:
        return int(match.group(1))
    return -1

def get_sorted_files(directory):
    """Get alphabetically sorted files from directory"""
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    return sorted(files, key=extract_number)

# Dataset class
class SpeakerDataset(Dataset):
    def __init__(self, data_dir, target_speaker):
        self.data = []
        self.labels = []
        
        # Load all positive samples from target speaker
        pos_dir = os.path.join(data_dir, target_speaker)
        pos_files = get_sorted_files(pos_dir)
        for file in pos_files:
            self.data.append(os.path.join(pos_dir, file))
            self.labels.append(1)
        print(f"Target Speaker Directory: {pos_dir}")
        print(f"Total Positive Samples (Class 1): {self.labels.count(1)}")

        # Get list of all speakers and their corresponding WAV files
        speakers = sorted([s for s in os.listdir(data_dir) if s != target_speaker])
        print(f"Other speakers: {speakers}")

        samples_per_negative = self.labels.count(1) // len(speakers)
        print(f"Samples per negative speaker: {samples_per_negative}")

        def generate_speaker_indices(wav_files, num_speakers):
            """
            Generate indices for each speaker's negative samples, maintaining alphabetical order
            """
            total_files = len(wav_files)
            files_per_speaker = total_files // (num_speakers - 1)
            
            speaker_indices = []
            start_idx = 0
            
            for i in range(num_speakers - 1):
                if i < num_speakers - 2:
                    end_idx = start_idx + files_per_speaker
                    indices = list(range(start_idx, end_idx))
                else:
                    indices = list(range(start_idx, total_files))
                speaker_indices.append(indices)
                start_idx = end_idx
            
            return speaker_indices
        
        # Process negative samples
        for speaker_idx, speaker in enumerate(speakers):
            neg_dir = os.path.join(data_dir, speaker)
            wav_files = get_sorted_files(neg_dir)  # Get alphabetically sorted files
            
            # Generate indices for current speaker
            speaker_indices = generate_speaker_indices(wav_files, len(speakers) + 1)
            indices = speaker_indices[speaker_idx]
            
            # Limit to samples_per_negative if specified
            if samples_per_negative > 0:
                indices = indices[:samples_per_negative]
            
            print(f"Speaker: {speaker}, using indices: {indices}")
            print(f"Files selected for {speaker}:")
            
            # Add selected files to dataset
            for idx in indices:
                if idx < len(wav_files):
                    file = wav_files[idx]
                    self.data.append(os.path.join(neg_dir, file))
                    self.labels.append(0)
                    print(f"Negative sample added: {os.path.join(neg_dir, file)}")

        # Print final dataset statistics
        print(f"\nFinal Dataset Statistics:")
        print(f"Total Positive Samples (Class 1): {self.labels.count(1)}")
        print(f"Total Negative Samples (Class 0): {self.labels.count(0)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data[idx]
        features = extract_mfcc_and_pitch(audio_path)
        label = self.labels[idx]
        return torch.FloatTensor(features), torch.LongTensor([label])
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        Early stopping class to stop training when validation loss stops improving.
        
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change in the validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter since we found an improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
        return self.stop_training

# Training function
def train_with_kfold(dataset, model_class, num_folds=0, num_epochs=0, batch_size=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_results = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'eers': [] 
    }
    
    # Variabel untuk menyimpan model terbaik
    best_model = None
    best_accuracy = 0.0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):

        model = model_class()
        model = model.to(device)

        train_subdata = Subset(dataset, train_idx)
        val_subdata = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subdata, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subdata, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        early_stopping = EarlyStopping(patience=5, delta=0)
        should_stop_training = False  # Flag untuk menghentikan training

        
        print(f"\nFold {fold}")
        best_fold_accuracy = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.squeeze())
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            avg_train_loss = train_loss/len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target.squeeze()).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            avg_val_loss = val_loss/len(val_loader)
            val_accuracy = correct/len(val_subdata)
            
            print(f'Epoch: {epoch+1}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')

            # Check early stopping conditions
            if avg_train_loss <= 0.001:
                print(f"Training loss is 0 at epoch {epoch+1}. Stopping training for this fold.")
                should_stop_training = True
            
            # Check EarlyStopping based on validation loss
            if early_stopping(avg_val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                should_stop_training = True
            
            if should_stop_training:
                break

        # Tambahkan perhitungan EER di sini
        eer, threshold = evaluate_model(model, val_loader, device)
        print(f'EER: {eer:.4f} at threshold: {threshold:.4f}')

        # Simpan hasil EER
        fold_results['eers'].append(eer)
            
        # Simpan model terbaik secara keseluruhan
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model.state_dict()
            
        # Simpan model terbaik per fold
        if val_accuracy > best_fold_accuracy:
            best_fold_accuracy = val_accuracy

            torch.save(model.state_dict(), f'output/best_model_fold_{fold}.pth')
        
        fold_results['train_losses'].append(train_loss/len(train_loader))
        fold_results['val_losses'].append(val_loss/len(val_loader))
        fold_results['val_accuracies'].append(val_accuracy)
        # fold_results['eers'].append(eer)  # Tambahkan ini

    print("\nK-Fold Cross-Validation Summary:")
    print(f"Average Validation Accuracy: {np.mean(fold_results['val_accuracies']):.4f} ± {np.std(fold_results['val_accuracies']):.4f}")
    print(f"Average Validation Loss: {np.mean(fold_results['val_losses']):.4f} ± {np.std(fold_results['val_losses']):.4f}")
    print(f"Average EER: {np.mean(fold_results['eers']):.4f} ± {np.std(fold_results['eers']):.4f}")  # Tambahkan ini
    
    # Simpan model terbaik keseluruhan
    if best_model is not None:
        torch.save(best_model, 'output/best_overall_model.pth')
        print(f"\nBest overall model saved with accuracy: {best_accuracy:.4f}")
    
    return fold_results

def save_training_results(results, output_dir='output10'):
    """
    Simpan grafik hasil pelatihan ke dalam file.
    
    Args:
        results: Dictionary yang berisi metrik pelatihan.
        output_dir: Direktori tempat menyimpan grafik.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot dan simpan Training and Validation Loss
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Training Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results['val_accuracies'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_validation_metrics.png'))
    plt.close()  # Tutup plot untuk menghemat memori

    # Plot dan simpan EER
    plt.figure(figsize=(6, 5))
    plt.plot(results['eers'], label='EER')
    plt.title('Equal Error Rate (EER)')
    plt.xlabel('Fold')
    plt.ylabel('EER')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'eer_metrics.png'))
    plt.close()  # Tutup plot untuk menghemat memori

# Main execution
def main():
    # Hyperparameters
    batch_size = 16
    num_epochs = 30
    # num_folds = 10
    num_folds = 5

    # Initialize dataset
    dataset = SpeakerDataset(
        data_dir='/path/to/dataset',
        target_speaker='target speaker',
        )
    
    if not os.path.exists('output10'):
        os.makedirs('output10')
    
    # Jalankan K-Fold Cross-Validation
    results = train_with_kfold(
        dataset, 
        model_class=XVectorNet, 
        num_folds=num_folds, 
        num_epochs=num_epochs, 
        batch_size=batch_size
    )
    
    # Simpan grafik hasil pelatihan
    save_training_results(results, output_dir='output')

if __name__ == "__main__":
    main()
