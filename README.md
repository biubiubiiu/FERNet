# FERNet

A PyTorch implementation of _[WACV 2022 paper](https://ieeexplore.ieee.org/document/9706627), "Fast and Efficient Restoration of Extremely Dark Light Fields"_

**Important Note:** This repository is **not** the official implementation of the paper. For the official implementation, please refer to [this repository](https://github.com/MohitLamba94/DarkLightFieldRestoration).

## Quick Start

1. Download the [L3F dataset](https://mohitlamba94.github.io/L3Fnet/) and create a symlink to the project root. Ensure your directory structure resembles the following:
   
   ```
   |-- L3FNet
       |-- L3F-dataset
           |-- jpeg
               |-- train
               |-- test
       |-- train.py
       |-- eval.py
       ...
   ```

2. Modify `config.toml` for necessary setup:
   
   ```toml
   [model]
   resolution = 9

   [data]
   split = '20'
   cropped_resolution = 9
   ```

- `split`: the subset for training/evaluation/testing ('20', '50', or '100').

- `resolution`: angular resolution of the input image.

- `cropped_resolution`: angular resolution of the cropped LF image.

3. To start training or testing, execute the following commands:
   
   ```sh
   # training
   python train.py config.toml
   
   # testing
   python eval.py config.toml --ckpt ${CKPT_PATH}
   ```

## Discussions

**Restoration of LF image**

The LF image is split into non-overlapping 3x3 views for restoration. When the resolution is not divisible by 3, overlapped views are introduced, which would be restored twice. The authors of the paper claim to randomly select the restoration result in such cases. However, I've adopted an alternative approach by averaging the results in the overlapped areas.

**Metrics Computation**

For a dataset with $M$ LF images, I calculate the metrics on the $A \times A$ SAIs separately, and average all the $M \times A^2$ scores as the final result. I think the official implementation compute the metrics under the Macro-Pixel representation, but this is rather weird and uncommon.
