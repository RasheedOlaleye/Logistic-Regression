
An end-to-end machine learning project that predicts Titanic passenger survival using Logistic Regression, with an interactive **Streamlit** web app.

I use the classic Titanic dataset (via Seaborn) to predict whether a passenger survived the sinking based on features like age, sex, ticket class, and fare.


NaN counts AFTER cleaning:
pclass      0
sex         0
age         0
fare        0
embarked    0
sibsp       0
parch       0
survived    0
dtype: int64

Training set: (712, 7), Test set: (179, 7)
Any NaN in X_train? 0
Any NaN in y_train? 0

Training Logistic Regression model...

==================================================
MODEL EVALUATION
==================================================
Accuracy:  0.7989
Precision: 0.7797
Recall:    0.6667
F1 Score:  0.7188
ROC-AUC:   0.8519
>>>>>>> 1ff61a0 (Initial commit: Titanic Logistic Regression with Streamlit)
