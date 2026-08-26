#  ustomer Churn Prediction & Survival Analysis

End-to-end churn analytics project: EDA → ML classification → 
SHAP explainability → survival analysis → deployed Streamlit app.

## 🛠 Tech Stack
Python | Pandas | Scikit-learn | XGBoost | SMOTE | SHAP | 
Lifelines (Kaplan-Meier) | Streamlit

## 📊 Results
| Model | ROC-AUC | Recall (Churn) |
|-------|---------|----------------|
| Logistic Regression (baseline) | 0.822 | 74% |
| XGBoost (tuned)                | 0.816 | 66% |

## Key Insights
1. **Contract type is the #1 churn driver** — month-to-month customers 
   churn ~42% vs ~3% for two-year contracts
2. **First 6 months = danger zone** — new customers churn at ~58%, 
   6x higher than customers with 4+ years tenure
3. **Fiber optic paradox** — premium users churn FASTER than DSL users, 
   indicating a price-value perception problem
4. **Electronic check = risk signal** — manual monthly payment means 
   customers re-evaluate the service every month
5. **Median lifetime**: month-to-month ~27 months vs 72+ months contracted

## Business Impact
Flagged ~340 high-risk customers in the test set. A targeted retention 
campaign saving 30% of them protects an estimated ~$357K in lifetime revenue.

## 🖥 Live Demo 
<img width="832" height="901" alt="image" src="https://github.com/user-attachments/assets/4605e8aa-dc2a-4b63-8ce4-d34ae6705e66" />

## High Risk
 <img width="723" height="877" alt="image" src="https://github.com/user-attachments/assets/1542f2f7-f51d-4b5c-bf44-c9c098229238" />

## Low Risk 
 <img width="793" height="903" alt="image" src="https://github.com/user-attachments/assets/292c995c-b524-47d1-a69e-946af7c296a4" />
 

## Run Locally
pip install -r requirements.txt
streamlit run app/app.py
