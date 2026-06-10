# Hyperbolic Spatio-Temporal Frequency Adaptation for Next Location Prediction
In this work, we identify a critical limitation in existing frequency-domain next-location prediction models: neglecting spatio-temporal context leads to inaccurate spectral distributions of user behavior representations. Motivated by this observation, we propose the HSFA framework, which seamlessly integrates hyperbolic space with multi-scale frequency domain modeling. Specifically, HSFA features a spatio-temporal enhancement module that explicitly modulates check-in sequences using spatio-temporal features, enabling precise decoupling of stable, long-term preferences from abrupt, short-term behavioral changes. Furthermore, HSFA incorporates a target time alignment module that leverages hyperbolic isometric rotation to align users' historical preferences with the target temporal context. Extensive experiments across three real-world datasets demonstrate that HSFA consistently outperforms state-of-the-art methods in both prediction accuracy and training efficiency.

<!-- <img width="1269" height="582" alt="image" src="https://github.com/user-attachments/assets/6dcf95bd-017a-4601-be79-9a8ce7e2539f" /> -->



## 📦 Environment
1. Clone this repository to your local machine.

2. Install the enviroment by running
```bash
conda env create -f HSFA.yml
```

## 🔧Training HSFA
To train the HSFA model, run the following command:
```bash
python main.py --data_name TKY --pre_hsfa_num_bands 7 --pre_hsfa_low_band_count 5 --batch 128 --epoch 50 --cuda True --gpu_id 0
```
or
```bash
bash start.sh
```
