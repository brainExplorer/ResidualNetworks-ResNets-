# Residual Networks Classification with PyTorch

This project implements image classification on the **CIFAR-10** dataset using **Residual Networks (ResNet)** in PyTorch. It supports two modes:

- ✅ Custom ResNet implementation from scratch  
- ✅ Transfer learning using pretrained `ResNet18` from `torchvision.models`

---

## 🔧 Features

- Residual Block implementation
- Custom ResNet with 4 residual layers
- Transfer Learning with modifications for CIFAR-10
- Data augmentation for improved generalization
- GPU support (CUDA)
- Accuracy reporting on test data

---

## 📁 Project Structure

```
.
├── main.py            # Main training and evaluation script
├── README.md          # Project documentation
└── data/              # CIFAR-10 dataset will be downloaded here
```

---

## 🧠 Model Architectures

### Custom ResNet

The architecture consists of:
- Conv2D → BatchNorm → ReLU
- 4 residual layers (with increasing feature maps and downsampling)
- Adaptive Average Pooling
- Fully Connected layer for 10-class output

### Transfer Learning ResNet18

Modifications made:
- First `Conv2D` adapted to CIFAR-10 (kernel size = 3)
- `MaxPool` layer removed
- Final `FC` layer replaced for 10-class classification

---

## 📦 Requirements

Install the necessary libraries:

```bash
pip install torch torchvision tqdm
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/resnet-cifar10.git
cd resnet-cifar10
```

### 2. Train the Model

Open `main.py` and set:
```python
use_custom_resnet = True  # or False for transfer learning
```

Then run:

```bash
python main.py
```

---

## 📊 Sample Output

```bash
Epoch [1/1], Loss: 1.5503
Accuracy of the model on the test set: 76.43%
```

> Results vary depending on the model and number of epochs used.

---

## 📈 Future Improvements

- Add more training epochs and learning rate scheduling
- Save and load trained models
- Integrate TensorBoard or Weights & Biases for visualization
- Support for CIFAR-100 and other datasets

---

## 📚 References

- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

---

## 🧑‍💻 Author

Developed with ❤️ by Ümit YAVUZ.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
