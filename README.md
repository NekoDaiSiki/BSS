# BSS

## Environment

| Component | Version     |
| --------- | ----------- |
| Python    | 3.8.0       |
| PyTorch   | 1.8.2+cu111 |
| CUDA      | 11.1        |
| cuDNN     | 8.0.5       |
| NumPy     | 1.24.4      |

## Training

### NSC-BSM

```bash
python train.py --config configs/nsc-bsm/gauss25.yaml --gpu 0
python train.py --config configs/nsc-bsm/poi30.yaml --gpu 0
python train.py --config configs/nsc-bsm/sidd_raw.yaml --gpu 0
```

### BSS

```bash
python train.py --config configs/bss/gauss25.yaml --gpu 0
python train.py --config configs/bss/poi30.yaml --gpu 0
python train.py --config configs/bss/sidd_raw.yaml --gpu 0
python train.py --config configs/bss/sidd_srgb/apbsn_no_r3.yaml --gpu 0
```



## Testing

```bash
# Gaussian 25
python test.py --config configs/bss/gauss25.yaml --resume ../exp/BSS/gauss25/ckpts/59.pth --gpu 0

# Poisson 30
python test.py --config configs/bss/poi30.yaml --resume ../exp/BSS/poi30/ckpts/59.pth --gpu 0

# SIDD raw
python test.py --config configs/bss/sidd_raw.yaml --resume ../exp/BSS/sidd_raw/ckpts/39.pth --gpu 0

# SIDD sRGB
python test.py --config configs/bss/sidd_srgb/apbsn_no_r3.yaml --resume exp/bss/sidd_srgb/apbsn_no_r3/ckpts/39.pth --gpu 0
```
