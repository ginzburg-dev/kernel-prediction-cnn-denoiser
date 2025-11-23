# Kernel-Prediction CNN Denoiser

A PyTorch-based deep learning denoiser that uses **kernel prediction** —  
a technique where a neural network predicts **spatially-varying convolution kernels**  
to reconstruct a clean image from a noisy input.

---

## This repo includes:

- CNN architecture for kernel prediction
- Separate kernel-application module
- Patch-based training pipeline
- Loss functions & metrics
- Dataset loader for multi-channel images (EXR/PNG/etc.)
- Inference script and visualization utilities
