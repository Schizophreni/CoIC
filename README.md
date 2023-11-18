### A Real-world deraining demo

We tune DRSformer trained w/o and w/ CoIC on a pseudo real-world dataset SPAData for another 105k iters. Surprisingly, we observe that DRSformer w/ CoIC tuned on SPAData can efficiently deraining challenging images contaminated with intricate rains. Below are a demon comparison. This indicates that the proposed CoIC can help train powerful deraining models using *both synthetic and real-world datasets*.

![demo](figures/tune_real_comp.gif)

​                    **DRSformer w/o CoIC tuned on SPAData**                                        **DRSformer w/ CoIC tuned on SPAData**
