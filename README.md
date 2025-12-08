# Classifying the Intrinsic Value of S&P 500 Options Using Ensemble Learning

## Josh Marsico - ITCS 3156


## 1. Introduction

**1.1 Problem Statement:** Options trading accounts for billions of dollars in the daily market, yet
accurately predicting the movement of option premiums is still one of the more challenging tasks in
quantitative finance. Unlike stocks, which move primarily based on supply and demand, option prices are
derived from a combination of the underlying asset's price, time to expiration, and implied volatility. The
specific problem addressed in this project is the binary classification of option profitability: predicting
whether a specific option contract is currently "In-The-Money" (ITM) or "Out-Of-The-Money" (OTM)
based on its trading metrics. Essentially, buying an option on a stock, either a call or a put, allows you to
exercise the right to buy or sell that stock at the price you locked in at the time of the option. The goal is
to see whether a model can correctly classify whether or not an option is profitable based on the data at
the time of its purchase. This is different from a traditional backtester, which is a predictive system using
the same data to try and give insight into potential future option strategies.

**1.2 Motivation and Challenges:** This problem is significant because it mimics the decision-making
process of automated trading systems. If a model can accurately classify the profitability of an option
based on the data at the time, it suggests that machine learning could potentially "reverse engineer" the
market's pricing mechanisms without relying on complex closed-form equations like the Black-Scholes
model to find the fair price. Essentially just saying that the model doesn’t know the exact math, but it can
recognize the pattern and assign the correct label anyway. The primary challenge in this domain is noise;
financial data is stochastic and heavily influenced by many external factors, making it difficult to separate
signal from noise.

**1.3 Approach:** My approach utilizes supervised machine learning to classify options. Instead of
predicting the exact price (a regression task), which is more prone to high error rates, I switched to
looking at this as a classification task (ITM vs. OTM). I implemented a baseline linear model (Logistic
Regression) and compared it against a non-linear ensemble method (Random Forest) to test the hypothesis
that option pricing dynamics are inherently non-linear. I thought this was an interesting view I hadn’t
really heard of before now.

## 2. Data and Environment

**2.1 Data Source:** The dataset utilized for this project is the "SPY Daily EOD Options Quotes
(2020-2022)" obtained from Kaggle. This dataset provides end-of-day quotes for options on the SPY ETF,
which tracks the S&P 500 index. The raw dataset contained approximately 3.5 million rows, covering
diverse market conditions including the post-COVID bull market of 2021 and the bear market of 2022.

**2.2 Data Visualization & Analysis:** The initial dataset required significant filtering to isolate relevant
signals. A key analytical step was visualizing the distribution of "Moneyness" (the ratio of Stock Price to
Strike Price). As shown in the data sample below, the raw data included deep OTM and deep ITM options,
which were filtered to focus on the active trading range.

<img width="622" height="146" alt="image" src="https://github.com/user-attachments/assets/23f5245f-ffe5-4a21-98d9-35f28576f28b" />

**2.3 Preprocessing:** To prepare the environment for machine learning, I performed the following
preprocessing steps:

1. **Filtering:** I removed all rows where “Volume” was zero, since illiquid options do not reflect true
    market prices.
2. **Feature Engineering:** I calculated a "Moneyness" feature (Underlying Price / Strike Price) and
    filtered for values between 0.90 and 1.10. This focused the model on "At-The-Money" options,
    which are the most difficult to predict.
3. **Target Creation:** I created a binary target variable, where 1 represents an option where the
    Underlying Price > Strike Price (ITM), and 0 otherwise.
4. **Cleaning:** Columns containing string formatting (e.g., commas in volume data) were collected
    and converted to numeric types.

## 3. Method

To solve the classification problem, I implemented and compared two distinct algorithms:

**3.1 Algorithm 1: Logistic Regression (Baseline)** - I selected Logistic Regression as the baseline model.
It is a linear classifier that estimates the probability of a binary outcome using a sigmoid function.

**Rationale:** In finance, it is standard practice to start with a simple linear model. If a complex model
cannot beat Logistic Regression significantly, then the added complexity is unjustified. This model
assumes a linear relationship between input features (like Implied Volatility) and the target.

**3.2 Algorithm 2: Random Forest Classifier** - The primary model used was the Random Forest
Classifier, an ensemble learning method that constructs a multitude of decision trees during training.

**Rationale:** Option pricing is non-linear. Factors like "Time to Expiry" (DTE) do not affect price in a
straight line; rather, time decay accelerates as expiration approaches. Random Forest is ideal for capturing
these non-linear dependencies and interactions between features. I configured the model with 50
estimators and a maximum depth of 10 to prevent overfitting. This methodology mainly came from many
of the sources I found, which have shown that Random Forest does perform better than linear models on financial data.


## 4. Results

**4.1 Experimental Setup:** The cleaned dataset (525,849) was split into a training set (80%) and a testing
set (20%). The models were evaluated on Accuracy, Precision, Recall, and the F1-Score.

**4.2 Model Performance:**

● **Logistic Regression:** Achieved an accuracy of **96.59%**.

● **Random Forest:** Achieved an accuracy of **97.29%**.

While the baseline model performed well, the Random Forest model showed a clear improvement, which
confirms that ensemble methods are better suited for the non-linear nature of options data as suggested by
the sources I found.

**4.3 Confusion Matrix Analysis:** The Confusion Matrix below visualizes the performance of the Random
Forest model on the test data.

<img width="375" height="334" alt="image" src="https://github.com/user-attachments/assets/60833d40-3644-4c06-9d0a-fabb9b247a95" />


The matrix shows a very low False Positive rate (1,375 misses vs 62,609 correct rejections). This indicates
the model is highly conservative, it is rarely classifying a worthless or low value option as profitable. In a
trading context, I believe this is a desirable trait as it minimizes the risk of entering bad trades.

**4.4 Feature Importance Analysis:** To understand why the model made its predictions, I analyzed the
Feature Importance.

<img width="559" height="358" alt="image" src="https://github.com/user-attachments/assets/1f1e40df-1ec0-43de-8cf6-5720cc2a9a8e" />

As seen above, C_LAST **(** Option Price **)** was the most dominant predictor, followed by C_IV **(** Implied
Volatility **)**. As a reality check this does make sense and aligns with financial theory: options that are
already expensive are more likely to be In-The-Money. In addition to that, the fact that the model
identified Implied Volatility as the second most important factor shows it successfully learned that
volatility is a key driver of option premiums, independent of price.

## 5. Conclusion

**5.1 Concluding Remarks:** This project successfully demonstrated that machine learning classifiers can
replicate the logic of option pricing with high accuracy (>97%). By processing over 500,000 historical
trade records, the Random Forest model learned to distinguish between profitable and unprofitable options
based solely on market data, without being given the Black-Scholes formula or something similar.

**5.2 Challenges & Learnings:** The primary challenge was data quality. The initial raw dataset was
massive and contained text-based errors which initially caused the entire dataset to be filtered out multiple
times. Overcoming this required some trial and error and really narrowing down which columns I truly
needed for this project. I learned that in financial ML, "Data Cleaning" is often more critical than model
tuning. I also learned a lot about Random Forest and ensemble learning in general, and found many of the
papers and sources to be very interesting, as this is relatively similar to an internship project I did as a
high school student for a financial advising firm. In that project, I wrote software to implement a
traditional backtesting algorithm as opposed to using a machine learning approach.

**5.3 Future Work:** A limitation of this model that I see is that it treats all days equally. A future
improvement would be to train separate models for "High Volatility" (2020/2022) and "Low Volatility" (2021) 
environments to see if more specifically trained models perform better.

## 6. References

**[1] Kyle Graupe.** "SPY Daily EOD Options Quotes (2020-2022)." _Kaggle_ , Version 1, 2023,
https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022.

**[2] Zhang, Y., et al.** "Stock Prediction Analysis Based on Logistic Regression and Random Forest."
_International Journal of Science and Research_ , vol. 12, no. 6, 2024.

**[3] Khaidem, Luckyson, et al.** "Predicting the Direction of Stock Market Prices Using Random Forest."
_arXiv preprint arXiv:1605.00003_ , 2016.

**[4] Visser, G. C.** "Option Pricing Boosted by Machine Learning Techniques." _Erasmus School of
Economics_ , 2024.

