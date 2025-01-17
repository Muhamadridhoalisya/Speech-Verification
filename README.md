# Speech Verification System using X-Vector Architecture

A robust speaker verification system implemented in Python using PyTorch. The system uses X-Vector architecture with MFCC and pitch features for speaker verification tasks.

## Features

- X-Vector deep neural network architecture for speaker embedding
- Combined MFCC and pitch features for improved verification
- K-fold cross-validation for robust model evaluation
- Early stopping mechanism to prevent overfitting
- Automatic data balancing for positive and negative samples
- Comprehensive training visualization and model checkpoint saving

## Requirements

```
See environment.yml
```

## Project Structure

```
├── dataset/                                # Dataset directory
    ├── speaker1/                           # Target speaker directory
    └── speaker2/                           # Other speakers directory
    ....
    └── speakern/                           # Other n speakers directory
├── environment.yml                         # Environmet conda
├── inference.py                            # Testing and inference script
├── output/                                 # Output directory for models and visualizations
│   ├── best_model_fold_*.pth               # Best models for each fold
│   ├── best_overall_model.pth              # Best overall model
│   └── training_validation_metrics.png
├── preprocessing.py                        # Main training script
└── train.py                                # Main training script
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/muhamadridhoalisya/speech-verification.git
cd speech-verification
```

2. Install required packages:
```bash
conda env create -f environment.yml
```
3. Create folder dataset and place the dataset there with a different folder name:
```bash
mkdir dataset
```

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── speaker1/
│   ├── segment_1.wav
│   ├── segment_2.wav
│   └── ...
├── speaker2/
│   ├── segment_1.wav
│   ├── segment_2.wav
│   └── ...
└── ...
```

## Dataset guide
- Files must be in wav format
- Files are made in 1 complete file, not fragmented
- If not in 1 file, then please do a merge
- Samplerate: 16kHz, Bitrate: 256kbps
- Minimum duration is 24 minutes
- Negative dataset (non-target) should be sought that sounds similar to the sound of the positive dataset (target) so that the model is more accurate.

## Usage

### Training

1. Prepare your dataset following the structure above.

2. Update the data directory path in `train.py`:
```python
dataset = SpeakerDataset(
    data_dir='path/to/your/dataset',
    target_speaker='speaker_name'
)
```

3. Run the training script:
```bash
python train.py
```

### Testing/Inference

1. Place your test audio files in a test directory.

2. Update the model and test directory paths in `inference.py`:
```python
model_path = 'output/best_overall_model.pth'
audio_folder = 'path/to/your/test/folder'
```

3. Run the inference script:
```bash
python inference.py
```

The script will output predictions for each audio file in the format:
```
File: audio1.wav, Predicted Class: 1, Probability: 0.9876
```
Where:
- Predicted Class: 1 indicates target speaker, 0 indicates non-target speaker
- Probability: confidence score for the prediction (0-1)

## Model Architecture

### Feature Extraction
- MFCC (Mel-frequency cepstral coefficients) with 40 coefficients
- Pitch features using YIN algorithm
- Feature normalization and combination

### X-Vector Network
- Frame-level feature extraction using 1D convolutions
- Statistics pooling layer for temporal aggregation
- Segment-level feature extraction using fully connected layers
- Binary classification output

## Training Parameters

- Batch size: 16
- Number of epochs: 30
- Number of folds: 5
- Early stopping patience: 5
- Learning rate: Adam optimizer with default parameters
- Dropout rate: 0.45

## Model Evaluation

The system evaluates performance using:
- Validation accuracy
- Training and validation loss curves
- K-fold cross-validation for robust performance estimation

## Output

The training process generates:
- Best model weights for each fold
- Best overall model weights
- Training and validation metrics visualization

## My environment and hardware
- Conda 24.11.2
- Linux Ubuntu 24.04.1 LTS

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License



## Acknowledgments

- I utilize the ljspeech dataset as the negative dataset and my own dataset as the positive dataset (target).
- X-Vector architecture implementation based on the paper "X-vectors: Robust DNN Embeddings for Speaker Recognition"
- Librosa library for audio processing
- PyTorch framework for deep learning implementation
  
```
This model is very sensitive to noise. Different types and conditions of recording devices will cause high errors.
```
