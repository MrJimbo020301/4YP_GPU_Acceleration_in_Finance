![image](https://github.com/user-attachments/assets/0b23985a-4132-4a10-8f25-d3b136ec1f37)# 4YP: Hybrid Ensembles for Time-Series Forecasting and Systematic Trading

*A Fourth-Year Project (4YP) in Engineering Science, University of Oxford, showcasing how individual investors can combine deep-learning ensembles with confidence-aware trading rules to enhance portfolio returns.*

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Reproducing the Results](#reproducing-the-results)
3. [Project Overview](#project-overview)
4. [Objectives](#objectives)
5. [Key Contributions](#key-contributions)
6. [Directory Layout](#directory-layout)
7. [Citation](#citation)
8. [License](#license)

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

2. **Enter the desired matching folder**

   <img src="https://github.com/user-attachments/assets/0103a512-0594-49ef-8da9-d5b12daa4df0" width="550" alt="Folder Structure"/>

   The basket trading algorithm could be found in this folder: `Final_Final Deployment of Basket Trading_MoE Prediction with MoE Trading`

4. **Choose an asset class** (12 in total)

   <img src="https://github.com/user-attachments/assets/5fae96e1-a2e9-445a-bec7-0af5004081e0" width="550" alt="Asset Classes"/>

5. **Implement Automatic Algorithm Trading Workflow**

   <img src="https://github.com/user-attachments/assets/cba7fc8d-7008-4b1c-8f9c-6c49f6de646a" width="650" alt="Notebook Execution"/>

   1. Select the `.ipynb` file ending in `_Final.ipynb` or `_Final_Verified.ipynb`, then run **all cells sequentially**.
      For example, for the Hybrid model predictions with confidence voting trading strategy implementation on brent oil futures, firstly run `Brent Oil_MoE_Fluctuating_New_Final_Predictions.ipynb` snippet to produce price predictions from all models on a rolling basis.
      This process might take several hours and should not be disturbed. It is highly recommended to execute the code on a remote server, not on your PC device, as it is quite computationally intensive. As a reference, the author processed the entire price prediction            workflow in about 4 or 5 hours on an NVIDIA 4090 card on the remote server. 

   2. Then, to see all the plotted diagrams as a visualised way of how the Mixture-of-Expert model has been constructed, you can run the optional `Brent Oil_Final_MoE Fluctuating_New_All Curve Plottings_Final.ipynb` script.
   3. Afterwards, execute `Brent Oil_Hybrid Models_Naive Trading & Polynomial Fittings_Final_Verified.ipynb` for Naive Trading or `Brent Oil_New_Hybrid Models_MoE Trading Strategy_Final_Verified.ipynb` for Confidence-voting Trading.
      The results would be generated sequentially with each code block, with the trading log and plotted diagrams available for reference.


   The very similar procedure works for all the other 7 cases except the final deployment of basket trading. In this case, we will firstly adopt the workflow applied to the ``Hybrid model price prediction +  Confidence Trading `` combination with all the selected asset
   baskets. That is, follow the above procedure 1 to 4 first. Price prediction results in `.pt` or `pkl` files on the repository, polynomial fitting and gradient outcomes, and individual asset trading simulations in the corresponding scrip workspace should be prepared
   beforehand.

   Then, on each selected asset, at the end of the trading simulation script, such as `Citigroup_New_Hybrid Models_MoE Trading Strategy_Final_Verified.ipynb`, there is a helper function to generate the individual asset's trading simulations. Rename it when necessary.
 
   <img src="https://github.com/user-attachments/assets/724c4c3a-9bc2-463b-acb9-f998a3ef5f89" width="550" alt="Asset Classes"/>

   There should be three additional CSV files in the local repository of that subfolder (returns_xxx.csv, signals_xxx.csv and confidence_xxx.csv).
   
   <img src="https://github.com/user-attachments/assets/eac3f32b-e37e-48bd-8bb5-af0dcd07ac9a" width="550" alt="Asset Classes"/>

   Repeat this process for all the other assets you want to trade together. Copy and paste the `returns_xxx.csv`, `signals_xxx.csv` and `confidence_xxx.csv` files to the parent folder (Under `Final_Final Deployment of Basket Trading_MoE Prediction with MoE Trading`).

   <img src="https://github.com/user-attachments/assets/7bcb6e6d-06a3-4c81-8e27-f1b680a2a587" width="550" alt="Asset Classes"/>

   Finally, run `Basket Trading_Report Writing_Final_Verified_All Plots.ipynb` code snippet. The basket trading simulations on each for each investmentary asset would be shown in the outputs.


7. **Customise Your Asset (optional)**

   Adjust the `TRAIN_START`, `VAL_START`, and `TEST_START` and all the corresponding variables referred to dates for each notebook script.

   Meanwhile, change the corresponding CSV files with adapted names. You can upload the new Excel worksheets from your PC to the repository if needed. 

   All downstream computations, including forecasts, polynomial fits/derivatives, and trading results, will update automatically.

   Make sure all the new date information is retrieved and updated from the website to the corresponding Excel worksheets on the local machine.

   <img src="https://github.com/user-attachments/assets/3f8dacf8-82bf-47f2-9678-f9f7e41ea715" width="650" alt="Editable Dates"/><br>

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

Detailed metrics, figures, and P\&L curves live in the `Report_Written_Materials`, `Report Diagrams_All` folder or summarised in Tabular Summaries_Final Report.xlsx worksheet.

---

## Directory Layout

```
4YP_GPU_Acceleration_in_Finance/
├── Tabular Summaries_Final Report.xlsx           # All Generated Results from the Report in Compilation
├── Trading Strategy xxx Implementations_xxx Price Predictions/                  # <Jupyter Notebooks by Model × Strategy>
│   └── <Asset>/
│       ├── <Asset>_xxx_xx_(New)_xx_Final.ipynb
│       └── <asset>_xxx_xx_(New)_xx_Final_Verified.ipynb
── Final_Final Deployment of Basket Trading_MoE Prediction with MoE Trading/     # <Jupyter Notebooks For Basket Trading>
    └── Basket Trading_Report Writing_Final_Verified_All Plots.ipynb
    └── <confidence_xxx.csv>
    └── <signals_xxx.csv>
    └── <returns_xxx.csv>
│   └── <Asset>/
│       ├── <Asset>_xxx_xx_(New)_xx_Final.ipynb
│       └── <asset>_xxx_xx_(New)_xx_Final_Verified.ipynb
├── Report_Written_Materials    # Saved forecasts, back-tests, plots
├── Report Diagrams_All         # Saved diagrams 
└── README.md                # you-are-here
```

---

## Citation

> **Jin, Y.** (2025). *GPU Acceleration in Finance: Hybrid Ensembles for Time-Series Forecasting and Systematic Trading.* Fourth-Year Project, Department of Engineering Science, University of Oxford.

---

## License

This repository is released under the **MIT License**.
