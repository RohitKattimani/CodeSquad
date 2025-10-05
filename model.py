import pandas as pd
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

"""Load datasets"""

# Load the positive and negative drug interactions datasets from HODDI
neg = 'HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset/neg.csv'  # --> Change to your path
pos = 'HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset/pos.csv'  # --> Change to your path

dfn = pd.read_csv(neg)
dfp = pd.read_csv(pos)

# Load the dictionary of drugs from HODDI
dictionary = 'faers_xml_2025q2/XML/3_ADR25Q2.xml'        # --> Change to your path

df_dict = pd.read_csv(dictionary, sep=",", on_bad_lines="skip", engine="python")


# Drop unnecessary columns of the dataset
dfn.drop(['time', 'row_index', 'SE_above_0.9'], axis=1, inplace=True)
dfp.drop(['time', 'row_index', 'SE_above_0.9'], axis=1, inplace=True)

# Change the target label for negative drug interactions to 0 to
dfn.replace({'hyperedge_label': {-1: 0}}, inplace=True)

"""Splitting the dataset"""

# Split: 80% train, 20% test
# Split positives
dfp_train, dfp_test = train_test_split(dfp, test_size=0.20, random_state=42)
# Split negatives
dfn_train, dfn_test = train_test_split(dfn, test_size=0.20, random_state=42)

# Combine data for training and testing
train_combined = pd.concat([dfp_train, dfn_train]).sample(frac=1, random_state=42)
test_combined = pd.concat([dfp_test, dfn_test]).sample(frac=1, random_state=42)

"""Converting all DrugBank IDs to a list"""

def convert_string_to_list(s):
    if isinstance(s, str):
        evaluated = ast.literal_eval(s)
        if isinstance(evaluated, list):
            return evaluated
        else:
            return [str(evaluated)]
    elif isinstance(s, list):
        return s

# Apply the conversion
train_combined['DrugBankID'] = train_combined['DrugBankID'].apply(convert_string_to_list)
test_combined['DrugBankID'] = test_combined['DrugBankID'].apply(convert_string_to_list)

"""Drop the unknown classes in testing"""

# List of unknown classes provided when running previous models
unknown_ids = ['DB03862', 'DB04482', 'DB04920', 'DB11050', 'DB12366', 'DB13151', 'DB14693', 'DB15270', 'DB18046']

def has_unknown(drug_ids):
    return any(d in unknown_ids for d in drug_ids)

train_combined = train_combined[~train_combined['DrugBankID'].apply(has_unknown)].reset_index(drop=True)
test_combined = test_combined[~test_combined['DrugBankID'].apply(has_unknown)].reset_index(drop=True)

print(train_combined['DrugBankID'].explode().isin(unknown_ids).any())
print(test_combined['DrugBankID'].explode().isin(unknown_ids).any())
test_combined.shape

"""Stacking the Ensemble Model"""


Y_true = test_combined['hyperedge_label'].values

# Fitting the MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

# Transforming that data for our model
X_train = mlb.fit_transform(train_combined['DrugBankID']).astype(float)
Y_train = train_combined['hyperedge_label']

estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('cb', CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_seed=42))
]

model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(), 
    passthrough=True,  # pass original features to final_estimator
    cv=5 
)

model.fit(X_train, Y_train)


X_test = mlb.transform(test_combined['DrugBankID']).astype(float)
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]  # confidence score for label = 1


test_combined['predicted_label'] = Y_pred
test_combined['confidence_score'] = Y_prob

# Output results
results = test_combined[['report_id', 'DrugBankID', 'predicted_label', 'confidence_score']]
results.to_csv('results.csv', index=False)
results.tail()

print(classification_report(Y_true, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_true, Y_pred))
print("ROC AUC:", roc_auc_score(Y_true, Y_prob))

"""Check if the model is overfitting by comparing with training set accuracy"""

# Predict on training set
Y_train_pred = model.predict(X_train)

# Calculating the accuracy
train_accuracy = accuracy_score(Y_train, Y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# More detailed metrics
print(classification_report(Y_train, Y_train_pred))
print("Training Confusion Matrix:\n", confusion_matrix(Y_train, Y_train_pred))

"""Save the trained model for model deployment"""
# Save the trained model at the desired path
joblib.dump(model, 'C:/Users/Rohit/TestProject/modeldata/model.pkl')
joblib.dump(mlb, 'C:/Users/Rohit/TestProject/modeldata/mlb.pkl')
