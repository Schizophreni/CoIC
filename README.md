

### Harnessing Joint Rain-/Detail-aware Representations to Eliminate Intricate Rains (ICLR 2024 poster)

[Wu Ran](https://github.com/Schizophrenip), [Peirong Ma](https://github.com/MPR666), [Zhiquan He](https://github.com/hezhiquan), [Hao Ren](https://github.com/leftthomas), [Hong Lu](https://github.com/Helencn)

[paper download](https://openreview.net/pdf?id=pdJXYfJjz9)

---

> **Abstract**: *Recent advances in image deraining have focused on training powerful models on mixed multiple datasets comprising diverse rain types and backgrounds. However, this approach tends to overlook the inherent differences among rainy images, leading to suboptimal results. To overcome this limitation, we focus on addressing various rainy images by delving into meaningful representations that encapsulate both the rain and background components. Leveraging these representations as instructive guidance, we put forth a Context-based Instance-level Modulation (CoI-M) mechanism adept at efficiently modulating CNN- or Transformer-based models. Furthermore, we devise a rain-/detail-aware contrastive learning strategy to help extract joint rain-/detail-aware representations. By integrating CoI-M with the rain-/detail-aware Contrastive learning, we develop [CoIC](https://github.com/Schizophreni/CoIC), an innovative and potent algorithm tailored for training models on mixed datasets. Moreover, CoIC offers insight into modeling relationships of datasets, quantitatively assessing the impact of rain and details on restoration, and unveiling distinct behaviors of models given diverse inputs. Extensive experiments validate the efficacy of CoIC in boosting the deraining ability of CNN and Transformer models. CoIC also enhances the deraining prowess remarkably when real-world dataset is included.*

---

### A real-world deraining demo on RealInt

![demo](figures/tune_real_comp.gif)             *Left:* ***DRSformer w/o CoIC adding SPAData***                                           *Right:*  ***DRSformer w/ CoIC adding SPAData***

We also find that after further trained on SPAData, DRSformer w/o CoIC cannot well handle both synthetic and real-world rain. However, with the help of the CoIC, DRSformer can achieve much better performances. Below are quantitative results (PSNR only, full results can be seen in our revised paper):

| Methods            | Rain200L  | Rain200H  | Rain800   | DID-Data  | DDN-Data  | SPAData   |
| ------------------ | --------- | --------- | --------- | --------- | --------- | --------- |
| DRSformer w/o CoIC | 39.32     | 29.27     | 28.85     | 34.91     | 33.71     | 45.46     |
| DRSformer w/ CoIC  | **39.70** | **30.31** | **29.73** | **35.02** | **33.94** | **46.03** |

The results tabulated above demonstrate that **a pre-trained DRSformer model can simultaneously perform well on five synthetic datasets, as well as the the real-world SPAData, with the help of the proposed CoIC strategy**.

---

### Datasets

|    Dataset     |                           Rain200L                           |                           Rain200H                           | Rain800  |                           DID-Data                           |                          DDN-Data                          |                           SPAData                            |                           RealInt                            |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------: | :----------------------------------------------------------: | :--------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Download links | [link](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) | [link](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) | [link]() | [link](https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view) | [link](https://xueyangfu.github.io/projects/cvpr2017.html) | [link](https://mycuhk-my.sharepoint.com/personal/1155152065_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2F1155152065%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fdataset%2Freal%5Fworld%5Frain%5Fdataset%5FCVPR19) | [link](https://pan.baidu.com/s/1lPn3MWckHxh1uBYYucoWVQ#list/path=%2F) |

Since *Rain800* stacks rainy image and groundtruth together, one needs to first depart them.

Note that official *DID-Data* divides training set into three density levels, one needs to first merge them and re-order them.

For realint, since its official dataset contains more than 0.6M pairs, we do not change its file tree.

**If the above preeprocesses bother you, please directly download our well-organized data sources.**

Make sure the dataset folders organize as follows:

```
datasets|
        |-Rain200L
                  |-train
                         |-rain
                         |-norain
                  |-test 
                       |-rain
                       |-norain
        |-Rain200H ... (same as Rain200L for Rain200H,Rain800,DID-Data,DDN-Data)
        |-spa
        |-RealInt
```

----

### Training

1. Please download all datasets listed above and make sure they are well-organized
2. You can train benchmark models without or with CoIC following:





### Evaluation

Plase following instruction below to test the model:



We evaluate PSNR/SSIM referred to the Python code from this repository: https://github.com/leftthomas/Restormer

****

### More real-world deraining comparison

We provide more high-quality real-world deraining results to demonstrate the effectiveness of the proposed CoIC, while enables DRSformer to learn from both mixed synthetic datasets and real-world SPAData datasets.

![real_world](figures/more_real_world.png)

