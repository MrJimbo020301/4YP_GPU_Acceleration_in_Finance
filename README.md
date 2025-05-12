# 4YP: GPU-Accelerated Price-Forecasting & Systematic Trading Strategies

*A Fourth-Year Project (4YP) in Engineering Science, University of Oxford, showcasing how retail investors can combine deep-learning ensembles with confidence-aware trading rules to enhance portfolio returns.*

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Reproducing the Results](#reproducing-the-results)
3. [Project Overview](#project-overview)
4. [Objectives](#objectives)
5. [Key Contributions](#key-contributions)
6. [Directory Layout](#directory-layout)
7. [Data Sources](#data-sources)
8. [Citation](#citation)
9. [License](#license)

---

## Quick Start

> **Prerequisites**
>
> * Python **3.12.7**
> * CUDA **12.8**
> * PyTorch **2.5.0** (+ CUDA 12.1)

---

## Reproducing the Results

1. **Pick a model–strategy pair**
   The grid below lists every combination of *price-prediction* technique and *trading-strategy* advancement available.

   <img src="https://github.com/user-attachments/assets/d636012c-49e8-4514-8812-d36f92aaba84" width="650" alt="Model × Strategy Matrix"/>

2. **Enter the matching folder**

   <img src="https://github.com/user-attachments/assets/0103a512-0594-49ef-8da9-d5b12daa4df0" width="550" alt="Folder Structure"/>

3. **Choose an asset class** (12 in total)

   <img src="https://github.com/user-attachments/assets/5fae96e1-a2e9-445a-bec7-0af5004081e0" width="550" alt="Asset Classes"/>

4. **Open the notebook**
   Select the `.ipynb` file ending in `_Final.ipynb` or `_Final_Verified.ipynb`, then run **all cells sequentially**.

   <img src="https://github.com/user-attachments/assets/cba7fc8d-7008-4b1c-8f9c-6c49f6de646a" width="650" alt="Notebook Execution"/>

5. **Customise date cut-offs (optional)**

   Adjust the `TRAIN_START`, `VAL_START`, and `TEST_START` and all the corresponding variables referred to dates for each notebook script.

   All downstream computations, including forecasts, polynomial fits/derivatives, and trading results, will update automatically.

   Make sure all the new date information is retrieved and updated from the website to the corresponding Excel worksheets on the local machine.

   <img src="https://github.com/user-attachments/assets/d2b0186c-2232-45f1-b563-0f0d78c2d481" width="650" alt="Editable Dates"/>

---

## Project Overview

Deploying deep-learning models for trading typically demands premium data, GPUs, and strong cross-disciplinary knowledge—placing the practice out of reach for many retail investors.
This project demonstrates that, with **public data** and **open-source tools**, individuals can:

* **Forecast prices** with state-of-the-art hybrid ensembles.
* **Trade systematically** using a transparent, confidence-aware rule set.
* **Outperform naïve baselines** on multiple asset classes.

Full methodology and results are documented in **`4YP_Report_Yuzhe_Jin_Final.pdf`**.

---

## Objectives

### 1 · Price-Prediction Progression

| Level                       | Approach                     | Purpose                         |
| --------------------------- | ---------------------------- | ------------------------------- |
| **Baseline**                | ARIMA                        | Classical statistical benchmark |
| **Selected Models**         | Ensemble of 9 ML forecasters | Combine model strengths         |
| **Mixture of Experts**      | Gated ensemble               | Dynamic regime selection        |
| **Hybrid MoE + Top Models** | MoE + best-performers        | Fuse filtering & averaging      |

### 2 · Trading-Strategy Progression

| Level                   | Strategy                  | Position Rule                          |
| ----------------------- | ------------------------- | -------------------------------------- |
| **Conservative**        | Naïve Long/Short          | Trade only if **all** models agree     |
| **Optimistic**          | Naïve Long/Short          | Trade if **any** model suggests a move |
| **Confidence-Weighted** | Composition of Confidence | Dynamic per-model trust                |

---

## Key Contributions

* **20 % gain** in predictive accuracy from the Hybrid MoE relative to single-model baselines.
* **≥ 70 % trend-matching accuracy** for Gradient Boosting, GRU, SVM, and N-BEATS across assets.
* **Up to 25 % higher monthly returns** than naïve trading rules using confidence weighting.
* **Positive returns in 65 % of scenarios**, beating UK risk-free rates by **6.67 % p.m.**

Detailed metrics, figures, and P\&L curves live in the `results/` folder.

---

## Directory Layout

```
4YP_GPU_Acceleration_in_Finance/
├── data/                    # CSV downloads per asset
├── notebooks/               # <Jupyter Notebooks by Model × Strategy>
│   └── <model>_<strategy>/
│       ├── <asset>_Final.ipynb
│       └── <asset>_Final_Verified.ipynb
├── results/                 # Saved forecasts, back-tests, plots
├── requirements.txt
└── README.md                # you-are-here
```

---

## Citation

> **Jin, Y.** (2025). *GPU Acceleration in Finance: Hybrid Ensembles for Time-Series Forecasting and Systematic Trading.* Fourth-Year Project, Department of Engineering Science, University of Oxford.

---

## License

This repository is released under the **MIT License**. See `LICENSE` for full terms.
