## Instructions for Reproducing or Duplicating the Results:
Pre-requisites:
Python 3.12.7
CUDA Version: 12.8 
Torch Version: 2.5.0
Torch Cuda Version: 12.1

For each combination of price prediction techniques and trading strategy advancements below, 
![image](https://github.com/user-attachments/assets/d636012c-49e8-4514-8812-d36f92aaba84)


Navigate to the corresponding folder shown below. 

![image](https://github.com/user-attachments/assets/0103a512-0594-49ef-8da9-d5b12daa4df0)


There are a total of 12 asset classes selected. Navigate in the subfolder as you want. 
![image](https://github.com/user-attachments/assets/5fae96e1-a2e9-445a-bec7-0af5004081e0)

For each selected asset in whichever combination of price prediction or trading techniques, select the Jupyter Notebook script (.ipynb) with either "_Final.ipynb" or "_Final_Verified.ipynb" suffix script files. 


Run the Jupyter Notebook file sequentially, the original version is expected to be executed successfully. 
![image](https://github.com/user-attachments/assets/cba7fc8d-7008-4b1c-8f9c-6c49f6de646a)


To migrate to the user's own cases with different train, validation and test cutoffs, they could directly alter the dates in each script to do so. This includes all related computations we would cover: price predictions,
polynomial fittings and polynomial derivatives results and plots, and the corresponding trading strategy implementations results and plots. All the dates should be matched with the Excel Worksheet. 
![image](https://github.com/user-attachments/assets/d2b0186c-2232-45f1-b563-0f0d78c2d481)


To choose an individual selected assets at various dates, the author has retrived the information from this website https://www.investing.com/. We download the correpsponding historical data into our local machine environment, such as from https://www.investing.com/equities/citigroup-historical-data. 

The three CSVs in the above should be updated accordingly. 

![image](https://github.com/user-attachments/assets/741c9ab0-8b44-4b73-aeaa-11f8bfc86b3a)




### The Original Complete Report Could be Read from the attached PDF files named "4YP_Report_Yuzhe_Jin_Final.pdf".

## Project Overview

Despite these developments mentioned above, individual investors may still find it challenging when seeking to harness Artificial Intelligence (AI) for trading. Implementing deep-learning models effectively not only requires quality datasets and computational resources but also a solid foundation of financial-markets and machine-learning principles. In addition, the complexity of the ML models can pose a barrier to entry for those without a technical background.

Hence, our Fourth Year Project (4YP) aims to bridge this gap by exploring how individuals can leverage current deep-learning models, incorporating comprehensive and understandable trading strategies to achieve investment returns for personal finance purposes. By focusing on accessible, open-source information available online (![orig-website-info][fig1])[^1], individual investors could potentially achieve a reasonable return on their portfolio management efforts, as shown in the monthly Profit and Loss (P&L) diagrams below (![Trading Outcome Example][fig2]).

---

## Project Objectives

To obtain the results presented above, this research project breaks down the overall problem into two interconnected areas: **time-series price prediction** and **quantitative trading-strategy design**.  

1. **Price Prediction Models Progressions**  
   We analyse a series of time-series forecasting approaches with increasing complexity:
   1. **Baseline Model (ARIMA)** – A classical statistical forecasting model as a performance baseline.  
   2. **Selected Model Combinations** – An ensemble of nine widely used machine-learning models to leverage their complementary strengths.  
   3. **Mixture of Experts (MoE)** – A gated ensemble that dynamically learns to weight or select among expert models.  
   4. **Hybrid MoE and Selected Models** – Combines the MoE architecture with the best-performing models to capture both dynamic regime filtering and model-averaging advantages.

2. **Trading Strategy Development Progression**  
   In parallel, we design trading strategies to adopt Long (buy) or Short (sell) positions based on our price predictions:
   1. **Naïve Conservative Strategy** – A cautious baseline that takes a position only when _all_ models predict a favourable trend.  
   2. **Naïve Optimistic Strategy** – Goes long if _any_ model predicts an increase, and short if _any_ model predicts a decrease.  
   3. **Composition of Confidence Strategy** – An advanced, confidence-weighted voting mechanism that dynamically adjusts each model’s trust over time.

---

## Project Contributions

The key contributions and quantitative achievements of this project are summarised as follows. Complete project folder including the ReadMe file and code scripts is on GitHub: [4YP_GPU_Acceleration_in_Finance](https://github.com/MrJimbo020301/4YP_GPU_Acceleration_in_Finance).

- Developed and evaluated an advanced ensemble forecasting framework, demonstrating that the Hybrid Mixture-of-Experts method improved predictive accuracy by approximately **20%** compared to individual models.  
- Identified Gradient Boosting, GRU, SVM, and N-BEATS as consistently high-performers, achieving trend-matching accuracy above **70%** across multiple asset classes.  
- Implemented a confidence-weighted trading strategy, resulting in simulated monthly returns up to **25%** higher than baseline naïve strategies.  
- Validated the practicality of the methodology, achieving positive simulated returns in over **65%** of scenarios—outperforming the bank’s risk-free savings rate by **6.67%** per month.


