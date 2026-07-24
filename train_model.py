"""
Step 3: Data Preprocessing & Logistic Regression Training
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# ---- 1. LOAD DATA ----
print("Loading Titanic dataset...")
df = sns.load_dataset('titanic')

# ---- 2. SELECT FEATURES ----
features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'sibsp', 'parch']
target = 'survived'

df = df[features + [target]].copy()

print("\nNaN counts BEFORE cleaning:")
print(df.isnull().sum())

# ---- 3. HANDLE MISSING VALUES ----
print("\nCleaning data...")

# Impute numeric columns
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())

# Impute categorical columns
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop any rows where target is missing
df = df.dropna(subset=[target])

# ---- 4. ENCODE CATEGORICALS ----
print("Encoding categorical variables...")

# Manual mapping is safer than LabelEncoder for features
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# For embarked, use LabelEncoder but handle unknowns
le_embarked = LabelEncoder()
df['embarked'] = le_embarked.fit_transform(df['embarked'])

# Save encoders
joblib.dump(le_embarked, 'le_embarked.pkl')
# For sex, we just used a dict mapping, so no encoder file needed

print("\nNaN counts AFTER cleaning:")
print(df.isnull().sum())

# ---- 5. FINAL SAFETY: DROP ANY REMAINING NaNs ----
before_drop = len(df)
df = df.dropna()
after_drop = len(df)
if before_drop != after_drop:
    print(f"\n⚠️  Dropped {before_drop - after_drop} rows with remaining NaN values")

# ---- 6. SPLIT DATA ----
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Any NaN in X_train? {X_train.isnull().sum().sum()}")
print(f"Any NaN in y_train? {y_train.isnull().sum()}")

# ---- 7. SCALE FEATURES ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

# ---- 8. TRAIN MODEL ----
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ---- 9. PREDICT ----
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ---- 10. EVALUATE ----
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ---- 11. CONFUSION MATRIX ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\n✅ Confusion matrix saved")
plt.show()

# ---- 12. ROC CURVE ----
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
print("✅ ROC curve saved")
plt.show()

# ---- 13. FEATURE IMPORTANCE ----
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)
print(feature_importance.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Coefficient', y='Feature', palette='RdBu_r')
plt.title('Feature Importance')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\n✅ Feature importance saved")
plt.show()

# ---- 14. SAVE MODEL ----
joblib.dump(model, 'logistic_model.pkl')
print("\n💾 Model saved to 'logistic_model.pkl'")
print("💾 Scaler saved to 'scaler.pkl'")
print("\n🎉 Done! Run: uv run streamlit run app.py")