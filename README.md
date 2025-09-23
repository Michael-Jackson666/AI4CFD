# AI4CFD: AI Methods for Computational Fluid Dynamics and PDE Solving

This repository collects various state-of-the-art artificial intelligence methods for solving Partial Differential Equations (PDEs), with a focus on computational fluid dynamics applications. Each method is implemented with comprehensive tutorials and complete training code.

## 🎯 Overview

Modern AI techniques have revolutionized the field of PDE solving, offering new approaches to tackle complex fluid dynamics problems. This repository serves as a comprehensive collection of:

- **Physics-Informed Neural Networks (PINNs)** - Neural networks that incorporate physical laws
- **Deep Operator Networks (DeepONet)** - Learning operators between function spaces  
- **Fourier Neural Operators (FNO)** - Neural operators in frequency domain
- **Transformer-based approaches** - Sequence-to-sequence models for PDE solving

## 📁 Repository Structure

```
AI4CFD/
├── PINNs/                 # Physics-Informed Neural Networks
│   ├── tutorial.ipynb     # Interactive tutorial
│   ├── train.py          # Complete training code
│   └── README.md         # Method-specific documentation
├── DeepONet/             # Deep Operator Networks
│   ├── tutorial.ipynb    
│   ├── train.py         
│   └── README.md        
├── FNO/                  # Fourier Neural Operators
│   ├── tutorial.ipynb   
│   ├── train.py        
│   └── README.md       
├── Transformer/          # Transformer-based methods
│   ├── tutorial.ipynb   
│   ├── train.py        
│   └── README.md       
└── utils/               # Shared utilities and helper functions
    ├── data_utils.py    # Data loading and preprocessing
    ├── plotting.py      # Visualization utilities
    └── metrics.py       # Evaluation metrics
```

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Michael-Jackson666/AI4CFD.git
   cd AI4CFD
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the methods:**
   - Start with any `tutorial.ipynb` notebook for interactive learning
   - Use `train.py` scripts for full training pipelines
   - Refer to method-specific READMEs for detailed explanations

## 📚 Methods Overview

### Physics-Informed Neural Networks (PINNs)
Learn how to incorporate physical laws directly into neural network training through automatic differentiation.

### Deep Operator Networks (DeepONet)
Discover operator learning techniques that can map between infinite-dimensional function spaces.

### Fourier Neural Operators (FNO)
Explore frequency-domain neural operators that achieve excellent performance on periodic and quasi-periodic problems.

### Transformer-based Approaches
Investigate how sequence-to-sequence models can be adapted for PDE solving tasks.

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.12+
- TensorFlow 2.9+ (for some implementations)
- Jupyter Notebook
- Scientific computing libraries (NumPy, SciPy, Matplotlib)

See `requirements.txt` for complete dependency list.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Whether it's adding new methods, improving existing implementations, or enhancing documentation, your contributions help make this resource better for everyone.

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue or contact the maintainers.

---
*This repository aims to bridge the gap between theoretical advances in AI for PDE solving and practical implementation.*
