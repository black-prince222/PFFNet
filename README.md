### PFFNet: A point cloud based method for 3D face flow estimation
Created by Dong Li,Yuchen Deng, Zijun Huang from GuangDong University of Technology.

### Citation
If you find our work useful in your research, please cite:

        @article{li2025pffnet,
            title={PFFNet: A point cloud based method for 3D face flow estimation},
                author={Li, Dong and Deng, Yuchen and Huang, Zijun},
                journal={Journal of Visual Communication and Image Representation},
                pages={104382},
                year={2025},
                publisher={Elsevier}
                }

### Abstract

In recent years, the research on 3D facial flow has received more attention, and it is of great significance for related research on 3D faces. Point cloud based 3D face flow estimation is inherently challenging due to non-rigid and large-scale motion. In this paper, we propose a novel method called PFFNet for estimating 3D face flow in a coarse-to-fine network. Specifically, an adaptive sampling module is proposed to learn sampling points, and an effective channel-wise feature extraction module is incorporated to learn facial priors from the point clouds, jointly. Additionally, to accommodate large-scale motion, we also introduce a normal vector angle upsampling module to enhance local semantic consistency, and a context-aware cost volume that learns the correlation between the two point clouds with context information. Experiments conducted on the FaceScape dataset demonstrate that the proposed method outperforms state-of-the-art scene flow methods by a significant margin.

### Installation
    pip install -r requirement.txt


### Usage

#### TRAIN
    python train_facescape_normal.py config_train_facescape.yaml

#### TEST
    python evaluate_facescape_normal.py config_evaluate_facescape.yaml
