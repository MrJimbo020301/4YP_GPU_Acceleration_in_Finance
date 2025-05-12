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


