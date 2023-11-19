## Contents

>- [A real-world deraining demo](#a-real-world-deraining-demo)
>- [More real-world deraining comparison](#more-real-world-deraining-comparison)
>
>- [Training histories of all models](#training-histories-of-all-models)

---



### A real-world deraining demo

We tune DRSformer trained w/o and w/ CoIC on a pseudo real-world dataset SPAData for another 105k iters. Surprisingly, we observe that DRSformer w/ CoIC tuned on SPAData can efficiently deraining challenging images contaminated with intricate rains. Below are a demon comparison. This indicates that the proposed CoIC can help train powerful deraining models using *both synthetic and real-world datasets*.

![demo](figures/tune_real_comp.gif)  *The Left: * ***DRSformer w/o CoIC tuned on SPAData**                               *The Right: *  ***DRSformer w/ CoIC tuned on SPAData***

We also find that after tuned on SPAData, DRSformer w/o CoIC cannot well handle both synthetic and real-world rain. However, with the help of the CoIC, DRSformer can achieve much better performances. Below are quantitative results (PSNR only, full results can be seen in our revised paper):

| Methods            | Rain200L | Rain200H | Rain800 | Rain1200 | Rain1400 | SPAData |
| ------------------ | -------- | -------- | ------- | -------- | -------- | ------- |
| DRSformer w/o CoIC |          |          |         |          |          |         |
| DRSformer w/ CoIC  |          |          |         |          |          |         |

----

### More real-world deraining comparison





----

### Training histories of all models

Below we show the training loss curves of BRN, RCDNet, DGUNet, IDT, and DRSformer trained on mixed synthetic datasets. Additionally, training loss curves of DRSformer tuned on SPAData are also included.

<img src="figures/train_hist.pdf" width="800"/>

