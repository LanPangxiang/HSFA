# Hyperbolic Spatio-Temporal Frequency Adaptation for Next POI Recommendation
We propose HSFA, a Hyperbolic Spatio-temporal Frequency Adaptive framework for next POI recommendation. Specifically, we develop a spatiotemporal enhancement module that encodes spatial and temporal factors across multiple dimensions to augment the user behavior sequence. Building on this foundation, we further incorporate a frequency-aware module that adaptively identifies both users’ long-term, persistent interests and their short-term, abrupt interest changes. Finally, a target-time alignment module aligns historical preferences with the temporal context of the prediction target via geometric rotations in hyperbolic space. Extensive
experiments on three real-world datasets demonstrate that HSFA consistently outperforms state-of-the-art next POI recommendation methods.

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
