# 🧠 CIFAR-10 CNN Classifier with PyTorch

A custom convolutional neural network built from scratch using PyTorch to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Includes full training and validation pipelines, model checkpointing, data augmentation, and training visualization. This model achieved 87% accuracy on the test dataset.

---

## 📌 Features

- Custom CNN (`MyNet`) with 6 Conv layers + BatchNorm + Dropout
- Stratified train/validation split using `StratifiedShuffleSplit`
- AutoAugment + RandomCrop + ColorJitter for training data
- Training loop with loss and accuracy tracking
- Learning rate scheduler with `StepLR`
- Model checkpoint saving every 10 epochs
- Final evaluation on test set
- Plots: training vs. validation loss & accuracy

---

## 🏗️ Model Architecture

- **Conv Layers**: 6 total, with increasing depth from 8 to 256 channels
- **Pooling**: MaxPooling after selected blocks
- **BatchNorm**: Applied to first 5 Conv layers
- **Activation**: ReLU throughout
- **Classifier**: GlobalAvgPool → FC(512) → FC(10)
- **Dropout**: 0.5 before the final layer

---

## 🧪 Data Pipeline

- CIFAR-10 automatically downloaded with `torchvision`
- Data is split into:
  - **Train**: 80%
  - **Validation**: 20% (stratified)
  - **Test**: official test set
- Uses `AutoAugmentPolicy.CIFAR10`, `ColorJitter`, random cropping, flipping for training augmentation

---

## 🚀 How to Run

1. **Install dependencies**:

   ```bash
   pip install torch torchvision matplotlib numpy scikit-learn tqdm
   ```

2. **Train the model**:

   ```bash
   python3 MyNet.py
   ```

3. **Results**:
   - Model saved as `cifar_mynet_final.pt`
   - Checkpoints saved every 10 epochs
   - Plots saved:
     - `loss_fig.png`

---

## 📊 Visual Output

Example of saved plots:

### 🔻 Loss

![Loss Plot](loss_fig.png)

---

## 📁 Project Structure

```text
├── your_script.py                # Main script containing model, training, etc.
├── data/                         # CIFAR-10 data (auto-downloaded)
├── figures/                      # Loss and accuracy plots
├── cifar_mynet_final.pt         # Final trained model
├── cifar_mynet_epoch_*.pt       # Epoch checkpoints
└── README.md
```
