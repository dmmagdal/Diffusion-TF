# Notes on Diffusion Models and their training


The Diffusion model uses a neural network (usually a UNet). It uses the model to remove the noise that it adds to images at each timestep. The Diffusion model adding noise to an image is called forward diffusion. The process of removing that noise is backward diffusion (which is what the neural network performs). In the original code, the neural network is set to be the denoise function when sampling from the Diffusion model. 

Main changes I wish to implement:
1) Establish the workflow such that the Diffusion model parameters can be saved in either a json or pickle format
2) UNet model can be parameterized and created as a functional model or Keras model
3) The overall training process is easier to read and walk through. Maybe even put it in as a keras model