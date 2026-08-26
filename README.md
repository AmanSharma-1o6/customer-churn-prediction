# 📞 Customer Churn Prediction & Survival Analysis

End-to-end churn analytics project: EDA → ML classification → 
SHAP explainability → survival analysis → deployed Streamlit app.

## 🛠 Tech Stack
Python | Pandas | Scikit-learn | XGBoost | SMOTE | SHAP | 
Lifelines (Kaplan-Meier) | Streamlit

## 📊 Results
| Model | ROC-AUC | Recall (Churn) |
|-------|---------|----------------|
| Logistic Regression (baseline) | 0.84 | 79% |
| XGBoost (tuned)                | 0.85 | 79% |

## 🔑 Key Insights
1. **Contract type is the #1 churn driver** — month-to-month customers 
   churn ~42% vs ~3% for two-year contracts
2. **First 6 months = danger zone** — new customers churn at ~58%, 
   6x higher than customers with 4+ years tenure
3. **Fiber optic paradox** — premium users churn FASTER than DSL users, 
   indicating a price-value perception problem
4. **Electronic check = risk signal** — manual monthly payment means 
   customers re-evaluate the service every month
5. **Median lifetime**: month-to-month ~27 months vs 72+ months contracted

## 💰 Business Impact
Flagged ~340 high-risk customers in the test set. A targeted retention 
campaign saving 30% of them protects an estimated ~$357K in lifetime revenue.

## 🖥 Live Demo (screenshots)
| High Risk | Low Risk |
|-----------|----------|
| (paste screenshot) | (paste screenshot) |

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app/app.py
