# Fourier Neural Operators (FNO)

Fourier Neural Operators (FNO) are neural networks that learn operators between infinite-dimensional function spaces by operating in the Fourier domain. FNOs are particularly powerful for problems with periodic or quasi-periodic behavior and have shown remarkable efficiency and accuracy for solving parametric PDEs.

## 🎯 Key Concepts

### What is FNO?
FNO learns mappings between function spaces by:
1. **Fourier Transform**: Converting inputs to frequency domain
2. **Spectral Convolution**: Applying learnable linear operators in Fourier space
3. **Inverse Transform**: Converting back to spatial domain
4. **Local Transform**: Adding pointwise nonlinear transformations

### Architecture Components
```
Input → FFT → Spectral Convolution → IFFT → Pointwise → Output
```

Key features:
- **Global receptive field**: Each output depends on all inputs
- **Spectral bias**: Naturally captures global patterns
- **Resolution invariance**: Can train on one grid, test on another
- **Efficiency**: O(N log N) complexity due to FFT

### Mathematical Foundation
For a function u(x), FNO computes:
```
(Kₗ(u))(x) = σ(Wᵤ(x) + (K ∘ u)(x))
```
where:
- K is a spectral convolution operator
- W is a pointwise linear transformation
- σ is a nonlinear activation function

## 📁 Files in this Directory

- `tutorial.ipynb` - Interactive Jupyter notebook tutorial
- `train.py` - Complete training script for various problems
- `models.py` - FNO architectures and variants
- `layers.py` - Core FNO layers and spectral operations

## 🚀 Quick Start

### Running the Tutorial
```bash
jupyter notebook tutorial.ipynb
```

### Training a Model
```bash
python train.py --problem navier_stokes --epochs 500 --lr 0.001
```

### Available Problems
- `darcy` - Darcy flow with variable permeability
- `navier_stokes` - 2D Navier-Stokes turbulence
- `burgers` - 1D/2D Burgers' equation
- `wave` - Wave propagation
- `maxwell` - Maxwell's equations

## 📊 Key Advantages

### 1. Resolution Independence
- Train on 64×64 grid, test on 256×256
- Maintains accuracy across resolutions
- Enables super-resolution applications

### 2. Computational Efficiency
- O(N log N) per layer vs O(N²) for standard convolution
- Faster inference than traditional numerical solvers
- Excellent parallel scalability

### 3. Global Information Flow
- Each output depends on all inputs
- Captures long-range dependencies naturally
- No receptive field limitations

### 4. Spectral Accuracy
- High accuracy for smooth solutions
- Natural handling of periodic boundary conditions
- Excellent representation of wave-like phenomena

## 🔧 Implementation Details

### Spectral Convolution Layer
Core operation in Fourier domain:
```python
def spectral_conv(u, weights):
    # Transform to Fourier domain
    u_ft = torch.fft.rfft2(u, dim=(-2, -1))
    
    # Multiply by learnable weights (truncated to lower modes)
    out_ft = torch.einsum("bixy,ioxy->boxy", u_ft[..., :modes], weights)
    
    # Transform back to spatial domain
    out = torch.fft.irfft2(out_ft, s=(u.size(-2), u.size(-1)))
    return out
```

### FNO Architecture
```python
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        # Spectral convolution layers
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) 
            for _ in range(4)
        ])
        
        # Pointwise linear layers
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(4)
        ])
```

### Training Strategy
1. **Multi-scale training**: Start with coarse grids, progressively refine
2. **Mode scheduling**: Gradually increase spectral modes during training
3. **Curriculum learning**: Simple to complex PDEs
4. **Data augmentation**: Rotation, translation, scaling

## 🎯 Applications

### Fluid Dynamics
- **Navier-Stokes**: Turbulent flow simulation
- **Weather prediction**: Atmospheric modeling
- **Ocean currents**: Large-scale circulation

### Wave Phenomena
- **Seismic modeling**: Wave propagation in Earth
- **Acoustics**: Sound wave simulation
- **Electromagnetic**: Maxwell's equations

### Material Science
- **Phase transitions**: Alloy solidification
- **Crystal growth**: Microstructure evolution
- **Diffusion**: Mass transport in materials

### Climate Modeling
- **Global circulation**: Atmospheric dynamics
- **Carbon cycle**: Greenhouse gas transport
- **Ice sheet dynamics**: Glacial modeling

## 📈 Performance Characteristics

### Speed Comparisons (typical)
- **Traditional PDE solvers**: Hours to days
- **FNO**: Seconds to minutes
- **Speedup**: 100x - 1000x faster

### Accuracy
- **Relative L2 error**: 1-5% for most problems
- **Long-term stability**: Maintains accuracy over time
- **Generalization**: Works on unseen parameter ranges

### Memory Efficiency
- **Parameter count**: 1M - 10M parameters typical
- **Memory usage**: Scales with resolution and batch size
- **Gradient memory**: Efficient due to spectral operations

## 🔬 Advanced Features

### Multi-Resolution Training
```python
# Train on multiple resolutions simultaneously
for resolution in [64, 128, 256]:
    data = get_data(resolution)
    loss = model(data)
    loss.backward()
```

### Adaptive Spectral Modes
```python
# Dynamically adjust number of Fourier modes
if epoch % 100 == 0:
    model.increase_modes()
```

### Physics-Informed FNO
```python
# Add physics constraints in Fourier domain
def physics_loss(u_pred, pde_operator):
    residual = pde_operator(u_pred)
    return torch.norm(residual)
```

## 🔗 Variants and Extensions

### 1. Factorized FNO
- Separable spectral convolutions
- Reduced parameter count
- Better for high-dimensional problems

### 2. Geo-FNO
- Spherical harmonics for global problems
- Weather and climate applications
- Handles curved geometries

### 3. U-FNO
- U-Net style skip connections
- Better for multi-scale problems
- Improved gradient flow

### 4. FNO-3D
- Three-dimensional spectral convolutions
- Volumetric problems
- Atmospheric and oceanic modeling

## 📚 Mathematical Background

### Spectral Methods Theory
FNO is based on spectral methods, which represent functions using basis functions:
```
u(x) = Σₖ ûₖ φₖ(x)
```

For periodic domains, φₖ(x) = e^(ikx) (Fourier basis).

### Universal Approximation
**Theorem**: Under suitable conditions, FNO can approximate any continuous operator mapping between function spaces.

### Convergence Analysis
- **Spectral accuracy**: Exponential convergence for smooth solutions
- **Aliasing errors**: Controlled by number of modes
- **Stability**: Bounded operators ensure stable training

## 🔗 References

1. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

2. Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., ... & Anandkumar, A. (2021). Physics-informed neural operator for learning partial differential equations. arXiv preprint arXiv:2111.03794.

3. Tran, A., Mathews, A., Xie, L., & Ong, C. S. (2021). Factorized fourier neural operators. arXiv preprint arXiv:2111.13802.