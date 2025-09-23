# ManipTrans for GigaHands-BiHSO @HANDS2025 Workshop HO-Tracker Challenge

This repo extends [ManipTrans](https://github.com/ManipTrans/ManipTrans) to the [GigaHands](https://github.com/Kristen-Z/GigaHands) dataset as part of the [HANDS2025 HO-Tracker Benchmark](https://hands-workshop.org/challenge2025.html), focusing on **bimanual hands interacting with a single object**.  

For other cases (single-hand single-object, bimanual with two objects), please refer to the original [ManipTrans](https://github.com/ManipTrans/ManipTrans) codebase.  

---

## 📑 Table of Contents
1. [Installation](#installation)  
2. [Prerequisites](#prerequisites)  
3. [Usage](#usage)  
4. [Check Out Our Paper & Workshop](#check-out-our-paper)  
5. [License](#license)  

---

## 🛠️ Installation
<a id="installation"></a>

Please refer to the original ManipTrans codebase [installation guide](https://github.com/ManipTrans/ManipTrans?tab=readme-ov-file#%EF%B8%8F-installation).  

---

## 📋 Prerequisites
<a id="prerequisites"></a>

Follow the original ManipTrans codebase [prerequisites](https://github.com/ManipTrans/ManipTrans?tab=readme-ov-file#-prerequisites) to download:  
1. MANO parameters  
2. Imitator checkpoints  

<details>
<summary>Steps:</summary>

### Demo Data for `GigaHands-BiHSO`
1. Download the demo data (2 sequences) from [here](https://m-8f2a9c.56197.5898.data.globus.org/data/ssrinath/datasets/Action/brics-mini/public_data/gigadual_demo.zip) and extract it into the `data/gigadual_demo` directory.

2. The directory structure should look like this:
    ```
    data
    └── gigadual_demo
        └── 001
            ├── urdf
                ├── scan.obj
                ├── scan-decompose.obj
                └── scan.urdf
            └── 001_sv_dict.npy
        └── 002
            ├── urdf
                ├── scan.obj
                ├── scan-decompose.obj
                └── scan.urdf
            └── 001_sv_dict.npy
    ```

3. For the full training set, please refer to the official HO-Tracker dataset on [Hugging Face](https://huggingface.co/datasets/LiKailin/HO-Tracker).  
</details>

---

## ▶️ Usage
<a id="usage"></a>

Follow the original ManipTrans [usage guide](https://github.com/ManipTrans/ManipTrans?tab=readme-ov-file#%EF%B8%8F-usage) for training and testing policies in other scenarios (single-hand single-object, bimanual with two objects).  

### 🤲 ⚽ Training Bimanual Single-Object Policies

1. **Preprocessing**  

   Preprocess data for both hands:
   ```bash
   # for Inspire Hand
   python main/dataset/mano2dexhand.py --data_idx G%000 --side right --dexhand inspire --headless --iter 7000
   python main/dataset/mano2dexhand.py --data_idx G%000 --side left --dexhand inspire --headless --iter 7000
   # for other hands, replace `inspire` with the corresponding hand name
   ```

2. **Training**  

   Train bimanual policies:
   ```bash
   python main/rl/train.py task=ResDexHand dexhand=inspire side=BiHSO headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[G%000] rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth early_stop_epochs=1000 actionsMovingAverage=0.4 experiment=cross_G%000_inspire
   ```
   As with single-hand training, adjust the `early_stop_epochs` parameter depending on task complexity.  

3. **Testing**  

   Test the bimanual policy:
   ```bash
   python main/rl/train.py task=ResDexHand dexhand=inspire side=BiHSO headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=false dataIndices=[G%000] rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth actionsMovingAverage=0.4 checkpoint=runs/cross_G%000_inspire__xxxxxx/nn/cross_G%000_inspire.pth
   ```

---

## 📄 Check Out Our Paper & Workshop
<a id="check-out-our-paper"></a>

Please chekout our full HO-Tracker challenge at ICCV 2025 HANDS workshop [here](https://hands-workshop.org/).
The workshop is based on ManipTrans, OakInk-V2 and GigaHands. If you find this work useful, please consider citing:  

```bibtex
  @inproceedings{li2025maniptrans,
    title={Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning},
    author={Li, Kailin and Li, Puhao and Liu, Tengyu and Li, Yuyang and Huang, Siyuan},
    booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
    year={2025}
  }
```

```bibtex
@inproceedings{fu2025gigahands,
  title={GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities},
  author={Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17461--17474},
  year={2025}
}
```

```bibtex
@inproceedings{zhan2024oakink2,
  title={Oakink2: A dataset of bimanual hands-object manipulation in complex task completion},
  author={Zhan, Xinyu and Yang, Lixin and Zhao, Yifei and Mao, Kangrui and Xu, Hanlin and Lin, Zenan and Li, Kailin and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={445--456},
  year={2024}
}
```

---

## 📜 License
<a id="license"></a>

This codebase is released under the [GPL v3 License](LICENSE).  
