# Notes on Diffusion Models and their training


The Diffusion model uses a neural network (usually a UNet). It uses the model to remove the noise that it adds to images at each timestep. The Diffusion model adding noise to an image is called forward diffusion. The process of removing that noise is backward diffusion (which is what the neural network performs). In the original code, the neural network is set to be the denoise function when sampling from the Diffusion model. 


### Main changes I wish to implement:

1) Establish the workflow such that the Diffusion model parameters can be saved in either a json or pickle format

2) UNet model can be parameterized and created as a functional model or Keras model

3) The overall training process is easier to read and walk through. Maybe even put it in as a keras model


### Preliminary Results

Able to run on laptop or Desktop but in CPU only. The laptop has an i5-5300 (gen 5) CPU (2 core, 4 logical processors) and 8GB of RAM. The desktop has an i7-10700 (gen 10) CPU (8 cores, 16 logical processors), 16GB of RAM, and an Nvidia GeForce RTX 2060 SUPER with 8GB of VRAM. Running test.py (the program that breaks down the entire diffusion process from the original repo piece by piece) on the desktop would have completed in over 8 days for a single epoch (CPU only because the code would not fit on the GPU). On a Google Colab instance (12GB of RAM, 12GB VRAM on an Nvidia Tesla K80), the process would have take almost 9 hours for a single epoch (running on the GPU). This means it is ill advised to attempt training in a reasonable time unless a user had an Nvidia GeForce RTX 3090 GPU (which has 24GB of VRAM) or some high end server GPU (such as the Nvidia A100 or P100).

I will attempt to come back at a later time when I have the necessary hardware to run it on my own.

Update: Adding mixed precision did help load the model onto the desktop's GPU but still ran into OOM errors. On free instance of Google colab (T4 GPU), model was set to train for 4.5 hours per epoch but loss was NaN.