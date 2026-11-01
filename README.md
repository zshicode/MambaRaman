# DiffMambaRaman: Mamba-based diffusion model for element identification using Raman spectroscopy

This repository proposes diffusion model implemented by Mamba for element identification of minerals using Raman spectroscopy. Deep learning for Raman spectroscopy has been adopted for mineral classification (Liu et al., 2017; Zhang et al., 2020). On this basis, this repository aims to identify specific elements in minerals or other inorganic materials by adopting Mamba-based diffusion as a representation learning model. Some code of the Mamba model is from [mamba.py](https://github.com/alxndrTL/mamba.py) and [MambaStock](https://github.com/zshicode/Mambastock).

## Requirements

The code has been tested running under Python 3.9.12, with the following packages and their dependencies installed:

```
numpy==1.16.5
pandas==1.4.2
pytorch==1.7.1
sklearn==0.21.3
```

## Usage

```bash
python main.py --model FlowMatching
```

`args.model` can be either DDIM or FlowMatching (default).

Users can define the elements to be identified by modifying list `elem`.

## Datasets

Following previous research (Zhang et al., 2020) for mineral classification, this repo validates the models on the excellent unoriented Raman spectra data of RRUFF dataset. To fix the dimension of different spectra, each spectrum is interpolated in 100-1400cm-1 with 1024 points. The spectra data of the minerals with only one sample per mineral class are removed.

The data can be downloaded via this [link](https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip). The Raman spectra data are acquired by spectrometers with different excitation wavelength (532nm, 780nm, ...). Besides, The dataset includes the raw data and the data processed by baseline removal and cosmicray removal. This repo provides options for users to adopt processed data or raw data with different excitation wavelength.

```
parser.add_argument('--wavelength', type=str, default='780',
                    help='Excitation wavelength of data')
parser.add_argument('--raw', type=str, default='Processed',
                    help='Processed/RAW')  
```

Function `preprocess(args.raw,args.wavelength)` copies the selected data to folder `./RRUFF_data`. Then, the data can be accessed by function `read_data('./RRUFF_data')`. For convience, the data selected by default setting is saved in `780-process.csv`, and can be accessed by reading the CSV instead of reading all files in data folder.

## Models

### DDIM

Revisit the famous DDPM (denoising diffusion probabilistic model), let $\epsilon\sim N(0,1),\alpha_t+\beta_t=1$, where $\beta_t$ is set as linear increasement from 0.0001 to 0.02 during the iteration procedure. The forward iteration

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon$$

can be simplified in a one-step manner

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

where $\bar\alpha_t=\prod^{t}_{s=1}\alpha_s$. Let $\epsilon_t$ denotes a neural network, DDPM is trained by minimize the diffusion error

$$L_d=\|\epsilon-\epsilon_t(x_t)\|^2$$

Then, the backward iteration

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t(x_t)\right)+\sigma_t\epsilon$$

Usually, DDPM sets $\sigma_t=\sqrt{\beta_t}$.

To improve the efficiency of diffusion, DDIM (denoising diffusion implicit model) removes $\sigma_t\epsilon$ to solve backward iteration in a few-step manner. Replace $\epsilon$ in one-step forward equation by $\epsilon_t(x_t)$, we reconstruct $x_0$ as

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_t(x_t)}{\sqrt{\bar\alpha_t}}$$

Then

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}}\epsilon_t(x_t)$$

Our work adopts DDIM as a representation learning model rather than a generative model, thus we only need to compute $\hat x_0$ to minimize the reconstruction error along with the diffusion error.

### Flow matching

Compared with DDPM/DDIM, flow matching accelerates diffusion model while enhancing its robustness. The forward iteration of flow matching is defined by

$$x_t=(1-t)x_0+t\epsilon$$

where $t\sim U(0,1)$. Different from DDPM/DDIM, flow matching derives velocity field as $v=x_0-\epsilon$, and trains a neural network $v_t$ to fit velocity field by minimizing the diffusion error

$$L_d=\|v_t(x_t)-(x_0-\epsilon)\|^2$$

As

$$x_t=(1-t)x_0+t\epsilon=(1-t)x_0+t(x_0-v)=x_0-tv$$

$x_0$ can be reconstructed by

$$\hat x_0=x_t+tv_t(x_t)$$

### Diffusion for representation learning

In basic diffusion model, $x_0$ is defined as raw data, and $\hat x_0$ is defined as its generative sample. If we

1. integrate encoder-decoder framework to diffusion, i.e., adopting $x_0=Encoder(u_0)$ for raw data $u_0$, 
2. define $\hat x_0$ as the denoised representation for $x_0$ (as well as $u_0$), 
3. reconstruct $\hat u_0=Decoder(\hat x_0)$ and minimize the reconstruction error $L_r=\|u_0-\hat u_0\|^2$.

Then we obtain latent diffusion framework, and the downstream task can adopt $\hat x_0$ as a representation. The total loss is the sum of downstream task loss (i.e. classification loss $L_c$ in this task), reconstruction error $L_r$ and diffusion error $L_d$.

$$L=L_c+\lambda_1 L_r+\lambda_2 L_d$$

We set $\lambda_1=1,\lambda_2=0.1$.

In this work, the diffusion networks (i.e. $\epsilon_t$ in DDIM and $v_t$ in flow matching) are implemented using Mamba.

## Results

The identification AUC ranges from 0.75 to 0.95 on different elements. Most elements can achieve the AUC more than 0.8.

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1024,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='FlowMatching',
                    help='Diffusion Model')
parser.add_argument('--wavelength', type=str, default='780',
                    help='Excitation wavelength of data')
parser.add_argument('--raw', type=str, default='Processed',
                    help='Processed/RAW')                     

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## Citation

MambaStock adopted Mamba for stock prediction while achieving superior performance, and this repo is also inspired by MambaStock. If you refer or use this repo for spectral data (or other kinds of data), citing MambaStock paper is also welcomed.
```
@article{shi2024mamba,
  title={MambaStock: Selective state space model for stock prediction},
  author={Zhuangwei Shi},
  journal={arXiv preprint arXiv:2402.18959},
  year={2024},
}
```

## References

Liu et al. Deep convolutional neural networks for raman spectrum recognition: a unified solution. Analyst. 2017

Zhang et al. Tranfer-learning-based Raman spectra identification. J Raman Spectrosc. 2020
