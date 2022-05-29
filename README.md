# Diffusion-TF

Description: Implement the original Diffusion model [code](https://github.com/hojonathanho/diffusion) which is from one of the authors of the orignal [paper](https://arxiv.org/pdf/2006.11239.pdf). This repo attempts to break down the original author's code into an easy to understand format, using styles from Tensorflow 2.


### Related sources:

 - [Pytorch implementation](https://github.com/abarankab/DDPM) of Denoising Diffusion Probabilistic Models
 - OpenAI's [Guided-Diffusion](https://github.com/openai/guided-diffusion) GitHub repository, related to the [Diffusion Models Beat GANs on Image Synthesis paper](https://arxiv.org/pdf/2105.05233.pdf)
 - OpenAI's [Glide-text2im](https://github.com/openai/glide-text2im) GitHub repository (note that there is no code for training this model and all model weights available are for a smaller, less powerful version of what OpenAI presents in their [paper](https://arxiv.org/pdf/2112.10741.pdf))
 - OpenAI's [Improved-Diffusion](https://github.com/openai/improved-diffusion) GitHub repository, related to the [Improved Denoising Diffusion Probabilistic Models paper](https://arxiv.org/pdf/2102.09672.pdf)


### Notes

 - OpenAI's implementations of diffusion model stem from converting the orignal author's code from Tensorflow to PyTorch, then extending it to their respoective projects.
 - The best way to understand the code is to go through the folloing path:
    1) The original Denoising Diffusion Probabilistic Models paper.
    2) The GitHub code from the author Jonathan Ho.
    3) OpenAI's Improved Denoising Diffusion Probabilistic Models paper.
    4) OpenAI's Improved Diffusion GitHub code associated with the preceeding paper.
    5) OpenAI's Guided Diffusion GitHub code (this will be for learning how to train Diffusion models for different tasks such as image super-resolution or conditional class-based generation).
    6) OpenAI's Glide-text2im GitHub code. While OpenAI did not include the code to train the Glide-text2im model, it was recommended in an associated [GitHub issue](https://github.com/openai/glide-text2im/issues/7) on the repository to leverage training from OpenAI's Guided-Diffusion repo to come up with a close approximation. Note that Glide-text2im requires an understanding of the CLIP model (also from OpenAI).