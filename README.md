# HumanVLA

This repository contains the official implementation associated with the paper: <b>HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid</b>.

Advances in Neural Information Processing Systems 2024 (NeurIPS 2024)

[[arXiv](https://arxiv.org/abs/2406.19972)]

## Installation
The codebase is developed under python3.8. We recommend using conda to create the environment
```
conda create -n humanvla python==3.8
conda activate humanvla
```

Then, install the official IsaacGym simulator from [this website](https://developer.nvidia.com/isaac-gym).

Last, install other dependencies by 
```
pip install -r requirements.txt
```


## Data
We use HITR dataset in this work. Please download it from [this link](https://drive.google.com/drive/folders/1nKpy4bRy7QTg_bzCj4mYoaWTCnW8Cgi7).

To access assests of the HITR dataset, you need to download the [HSSD dataset](https://huggingface.co/hssd), replace `HSSD_ROOT` in ```data/cp_hssd.py``` and copy assets by 
```
python data/cp_hssd.py
```

The prepared data directory will have the following structure:
```
----data
    |---- motions
    |---- assets
    |---- HITR_tasks
    |---- humanoid_assets
```


## Play

Download our pretrained checkpoints from [this link](https://drive.google.com/drive/folders/1Sg73ooIpFFBK195QmCJ181bBuKFRqeiA).

Then you can play with the humanoid by 

```
# humanvla-teacher
python main.py --test --name play --force --num_envs 4 --cfg cfg/teacher_rearrangement.yaml --ckpt weights/humanvla_teacher.pth
# humanvla
python main.py --test --name play --force --num_envs 4 --cfg cfg/student_rearrangement.yaml --ckpt weights/humanvla.pth
```

Qualitative results are available in [./demo](./demo)

## Evaluation

You can evaluate the pretrained models with ```--eval``` args:
```
# humanvla-teacher
python main.py --test --eval --name play --force --cfg cfg/teacher_rearrangement.yaml --ckpt weights/humanvla_teacher.pth
# humanvla
python main.py --test --eval --name play --force --cfg cfg/student_rearrangement.yaml --ckpt weights/humanvla.pth
```

## Train

The training process of HumanVLA consists of two stages, i.e. state-based teacher learning in stage 1 and vision-language directed policy distillation in stage 2. 

Besides, the stage 1 consists of carry-curriculum pretraining and object rearrangement learning.

The holistic training process includes 3 subprocesses:
```
# carry
torchrun --standalone --nnode=1 --nproc_per_node=4 -m main --headless --cfg cfg/teacher_carry.yaml --name teacher_carry
# teacher
torchrun --standalone --nnode=1 --nproc_per_node=4 -m main --headless --cfg cfg/teacher_rearrangement.yaml --name teacher_rearrange
# humanvla
torchrun --standalone --nnode=1 --nproc_per_node=4 -m main --headless --cfg cfg/student_rearrangement.yaml --name student_rearrange
```

We use ```torchrun``` to enable DDP and launch training. You can modify GPUs settings according to your devices. 

The training results will be saved in ```./logs/[name]``` folder. Argument ```--force``` will override an existing folder.


## Misc
We provide codes for referencing the generation processes in ```./gen```, including generating motions, waypoints, features, ...

## Acknowledgement

We appreciate the efforts of following open-sourced works: [ASE](https://github.com/nv-tlabs/ASE), [HSSD](https://huggingface.co/hssd), [OMOMO](https://github.com/lijiaman/omomo_release) and [SAMP](https://samp.is.tue.mpg.de/), which help the development of our work.


## Citation
If you find our work useful, please cite:
@misc{xu2024humanvla,
      title={HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid}, 
      author={Xinyu Xu and Yizheng Zhang and Yong-Lu Li and Lei Han and Cewu Lu},
      year={2024},
      eprint={2406.19972},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2406.19972}, 
}