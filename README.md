#Energy-guided Denoising Contrastive Learning for Molecule Property Prediction


### Environment 

See [here](docs/env_setup.md) for setting up the environment.




To train on different splits like All and All+MD, we can follow the link above to download the datasets.


## Training ##



We train EDCL on the PCQ V2 dataset by running:
    python DeNoise_MPP/Denoise_Contrastive_MPP/main.py



## Acknowledgement ##

Our implementation is based on [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [e3nn](https://github.com/e3nn/e3nn), [timm](https://github.com/huggingface/pytorch-image-models), [ocp](https://github.com/Open-Catalyst-Project/ocp), [Equiformer](https://github.com/atomicarchitects/equiformer).