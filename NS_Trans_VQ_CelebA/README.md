<h1 align="center">
  <b>NS-VQ and TransVQ</b><br>
</h1>


##### This Repository is adapted from https://github.com/AntixK/PyTorch-VAE. Appreciate it!

**Update 11/3/2025:** Adjusted PyTorch Lightning import commands to support 2.5.2 version

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 (tested up to 2.5.2)
- All experiments were run on a CUDA-enabled GPU

### Installation
```
$ pip install -r requirements.txt        # --no-user if in virtual env
```

### Usage
```
# or run locally
$python run.py -c configs_rebuild/vq_vae_face_q_update_softupdate.yaml 
```
**Config file template - See folder "configs_rebuild"**
## Model Config Overview
```yaml
model_params:
  # ------------------------------
  # Model Architecture
  # ------------------------------
  name: "VQVAE"
  in_channels: 3
  img_size: 128
  embedding_dim: 64              # latent embedding dim
  num_embeddings: 1024           # codebook size
  hidden_dims: [128, 256]        # encoder/decoder widths
  vq_flag: True                  # enable VQ (True) or Vanilla AE (False)

  # ------------------------------
  # VQ-VAE / NS-VQ / TransVQ Settings
  # ------------------------------
  vq_parameters:
    # ------------------ Core VQ options ------------------
    vq_mod: vq                   # vq | ema | vq_new | simvq | transvq  
                                 # vq      = baseline STE VQ
                                 # ema     = VQ-VAE-2 EMA codebook
                                 # vq_new  = Suppl. Sec 10 update rule
                                 # simvq   = Simple VQ (our backbone)
                                 # transvq = TransVQ (ours)
    
    beta: 0.25                   # commitment loss weight
    alpha: 1.0                   # embedding loss weight

    decay: 0.9995                # EMA decay (only used if vq_mod = ema)

    # ------------------ NS-VQ switches ------------------
    q_flag: True                 # True  -> new STE (Sec 3.3.3 of paper)
    mem_flag: False              # True -> NS-VQ (kernel / memory update)
                                 # mem_flag=True + vq_mod=vq = NS-VQ

    soft_sample_flag: False      # soft code assignment instead of argmin

    # ------------------ Random exploration of codebook --------------
    random_flag: False
    random_prob: 0.0
    random_decay_method: "constant"   # "constant|linear|exp|cosine|step|inverse|poly|sigmoid|cyclic"
    
    # decay schedule for σ² in kernel update (NS-VQ)
    decay_kwargs:
      decay_rate: 0.99           # exp
      decay_factor: 0.5          # step
      step_size: 5               # step
      k: 10.0                    # poly / inverse
      sharpness: 0.5             # sigmoid
      center: 5                  # sigmoid center epoch
      num_cycles: 1              # cyclic

data_params:
  data_path: "/isilon/.../celeba_hq_256"
  train_batch_size: 16
  val_batch_size: 16
  patch_size: 64
  num_workers: 4

exp_params:
  manual_seed: 1265
  LR: 0.0005
  scheduler_gamma: 0.995
  weight_decay: 0.0
  kld_weight: 0.00025

trainer_params:
  accelerator: gpu
  devices: 1
  max_epochs: 300
  gradient_clip_val: 1.0

logging_params:
  save_dir: "logs/"
  name: "VQVAE_FACE_transVQ"

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