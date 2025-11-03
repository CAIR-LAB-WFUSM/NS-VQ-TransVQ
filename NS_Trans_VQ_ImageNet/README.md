<h1 align="center">
  <b>NS-VQ and TransVQ</b><br>
</h1>


##### This Repository is adapted from https://github.com/CompVis/latent-diffusion.git. Appreciate it!

**Update 11/3/2025:** Adjusted PyTorch Lightning import commands to support 2.5.2 version

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 (tested up to 2.5.2)
- All experiments were run on a CUDA-enabled GPU


### Usage
```
# or run locally
$python main.py --base configs/vqvae/vq-f4_new_update.yaml -t True --gpus 0 -l logs_new -n vq_new_exp_decay_50000iter_0.1r_0.01init_warmup_constance_1epoch_10peak --no-test true

```
**Config file template - See folder "configs"**
```
vq-f4_new_update.yaml: NS-VQ
vq_f4_simvq.yaml: simVQ
vq_f4_transvq.yaml: TransVQ
vq_f4.yaml: VQVAE
```
**View TensorBoard Logs**
```
$ cd logs/<experiment name>/version_<the version you want>
$ tensorboard --logdir .
```


### License
**Apache License 2.0**

| Permissions      | Limitations       | Conditions                       |
|------------------|-------------------|----------------------------------|
| ✔️ Commercial use |  ❌  Trademark use |  ⓘ License and copyright notice | 
| ✔️ Modification   |  ❌  Liability     |  ⓘ State changes                |
| ✔️ Distribution   |  ❌  Warranty      |                                  |
| ✔️ Patent use     |                   |                                  |
| ✔️ Private use    |                   |                                  |


### Citation
```
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}

$ This repository adapts AntixK's code for a specific medical imaging use case.
```