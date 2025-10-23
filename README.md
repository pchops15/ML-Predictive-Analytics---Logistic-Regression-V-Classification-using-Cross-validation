# ML-Predictive-Analytics---Logistic-Regression-V-Classification-using-Cross-validation
**LendingClub Loan Default Prediction**

This project applies machine learning techniques in R to predict loan default outcomes using real LendingClub data, replicating and extending the analytical approach taken by Emily Figel in the case study. The goal was to design a prediction tool that could help investors make more informed, risk-adjusted lending decisions based on borrower characteristics.

**Objective**

The primary objective was to identify which borrowers are most likely to default and to recommend an optimal predictive model for Figel’s investment team. The project emphasizes both technical modeling and business interpretability — ensuring results are actionable for decision-makers, not just statistically sound.

**Methods**

The analysis follows a structured workflow:

Data Cleaning & Preparation

- Filtered loan records to include only finalized outcomes: Fully Paid, Charged Off, and Default.
- Engineered key features such as fico (average of FICO range), income (handling joint vs. individual applications), and combined_dti.
- Removed incomplete observations and encoded categorical variables as factors.

Feature Selection

- Chosen predictors include borrower characteristics and loan attributes such as int_rate, loan_amnt, dti, fico, verification_status, purpose, and sub_grade.

Model Development

- Implemented and compared two supervised learning models in R:
  - Classification Tree (CART) using rpart  
  - Logistic Regression using glm
- Evaluated both models with 4-fold cross-validation to estimate out-of-sample performance.

Model Evaluation Metrics

- LogLoss – to measure probability calibration.
- Accuracy and Confusion Matrix – to assess classification performance on the 20% holdout set.
- Interpretability – qualitative evaluation based on clarity of decision rules and business usability.

**Results**

Logistic Regression outperformed the classification tree numerically, achieving:

LogLoss: ~0.339

Accuracy: ~87.4%

The Classification Tree achieved a LogLoss of ~0.376, but offered greater interpretability for non-technical users.

The logistic model’s coefficients confirmed expected financial behavior:

Higher FICO scores reduce the odds of default.

Higher debt-to-income ratios (DTI) increase default risk.

The interest rate coefficient showed strong inverse association, likely reflecting internal risk-based pricing.

**Key Insights**

Model Recommendation: Logistic regression was proposed as the ideal prediction tool, as it delivers more reliable probability estimates while maintaining transparency for business decision-making.

Interpretation Challenge: The classification tree failed to visualize rule-based paths due to class imbalance — predicting all loans as “Repaid.” This underscores the need for class balancing in future iterations.

Risk Perspective: The model enables Figel’s team to adjust investment strategies by setting probability thresholds according to their risk tolerance (e.g., risk-averse teams prioritize minimizing false negatives).

**Tools & Libraries**

Language: R

Key Packages: tidyverse, rpart, rpart.plot, MLmetrics, caret

Techniques: Data cleaning, feature engineering, classification modeling, cross-validation, performance evaluation.

📈 Business Impact

This project demonstrates how data-driven modeling can enhance investment decision-making in peer-to-peer lending. By quantifying borrower risk and calibrating default probabilities, Figel’s team can optimize portfolio returns while maintaining appropriate risk exposure.
