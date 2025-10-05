from catboost import CatBoostClassifier
import ast
import gradio as gr
import pandas as pd
import joblib
import os


# Loading trained model and mlb
model_path = 'modeldata/model.pkl'
mlb_path = 'modeldata/mlb.pkl'

model = joblib.load(model_path)
mlb = joblib.load(mlb_path)

# Loading datasets
neg = 'HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset/neg.csv'
pos = 'HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset/pos.csv'

dfn = pd.read_csv(neg)
dfp = pd.read_csv(pos)
dfn.replace({'hyperedge_label': {-1: 0}}, inplace=True)
df_combined = pd.concat([dfp, dfn]).sample(frac=1, random_state=42)

dictionary = 'C:/Users/Rohit/TestProject/HODDI/dataset/HODDI_v1/dictionary/Drugbank_ID_SMILE_all_structure links.csv'
df_dict = pd.read_csv(dictionary)
df_dict = df_dict.rename(columns={'DrugBank ID': 'DrugBankID'})

# Convert all strings to a list
def convert_string_to_list(s):
    if isinstance(s, str):
        evaluated = ast.literal_eval(s)
        return evaluated if isinstance(evaluated, list) else [str(evaluated)]
    elif isinstance(s, list):
        return s

df_combined['DrugBankID'] = df_combined['DrugBankID'].apply(convert_string_to_list)

# Drop unknown IDs
unknown_ids = ['DB03862', 'DB04482', 'DB04920', 'DB11050', 'DB12366', 'DB13151', 'DB14693', 'DB15270', 'DB18046']
df_combined = df_combined[~df_combined['DrugBankID'].apply(lambda x: any(d in unknown_ids for d in x))].reset_index(drop=True)

# Name-ID mappings
name_to_id = dict(zip(df_dict['Name'], df_dict['DrugBankID']))
id_to_name = dict(zip(df_dict['DrugBankID'], df_dict['Name']))

def extract_valid_drugs(interaction_df, drug_dict_df):
    all_interaction_ids = set(d for lst in interaction_df['DrugBankID'] for d in lst)
    valid_drugs = drug_dict_df[drug_dict_df['DrugBankID'].isin(all_interaction_ids)]
    id_to_name = dict(zip(valid_drugs['DrugBankID'], valid_drugs['Name']))
    name_to_id = dict(zip(valid_drugs['Name'], valid_drugs['DrugBankID']))
    drug_name_list = valid_drugs['Name'].unique().tolist()
    return id_to_name, name_to_id, drug_name_list

id_to_name, name_to_id, drug_list = extract_valid_drugs(df_combined, df_dict)

def prepare_model_input(drug_names):
    try:
        drug_ids = [name_to_id[name] for name in drug_names]
        return mlb.transform([drug_ids]).astype(float)
    except Exception as e:
        print(f"Input error: {e}")
        return None

# Risk mapping for explanation, we just need to connect this to the Reactome API for expansion
risk_mapping = {
    ("Warfarin", "Aspirin"): "Bleeding risk",
    ("Amiodarone", "Digoxin"): "Arrhythmia risk",
    ("Elgocalciferol", "Adapalene"): "Unknown risk – consult pharmacist",
    ("Ampicillin", "Methylphenidate"): "Possible CNS interactions"
}

def get_risk_description(drugs):
    return risk_mapping.get(tuple(sorted(drugs)), "Unknown risk – consult pharmacist")

# Prediction function
def predict_combination(drugs):
    if len(drugs) < 2:
        return "Please select at least 2 drugs", "0%", ""
    input_data = prepare_model_input(drugs)
    if input_data is None:
        return "Error preparing input", "0%", ""
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    confidence = max(proba) * 100
    if prediction == 1:
        result = "Risky"
        risk_info = get_risk_description(drugs)
    else:
        result = "Safe"
        risk_info = "No known adverse interaction"
    return result, f"{confidence:.0f}%", risk_info

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>💊 MedSafe 💊</h1>
        <p>
            Welcome to the MedSafe Platform! <br>
            Select multiple drugs to check for potential interactions. <br>
            The model predicts if the combination is safe or risky, and shows a confidence score.
        </p>
    </div>
    """)

    with gr.Row():
        drug_select = gr.Dropdown(choices=drug_list, label="Select your Drug Names", multiselect=True, interactive=True)

    with gr.Row():
        selected_display = gr.Textbox(label="Selected Drugs", interactive=False)
        output = gr.Textbox(label="Prediction Result")
        confidence = gr.Textbox(label="Confidence Score")
        risk_info_box = gr.Textbox(label="Risk Explanation")

    selected_drugs = gr.State([])

    def update_selection(drugs):
        return ", ".join(drugs) if drugs else "", drugs

    drug_select.change(fn=update_selection, inputs=drug_select, outputs=[selected_display, selected_drugs])

    submit_btn = gr.Button("Predict", variant="stop")
    submit_btn.click(fn=predict_combination, inputs=selected_drugs, outputs=[output, confidence, risk_info_box])

    clear_btn = gr.Button("Clear")
    clear_btn.click(fn=lambda: ("", [], [], "", "", ""), outputs=[selected_display, selected_drugs, drug_select, output, confidence, risk_info_box])

    gr.Markdown("""
    <div style="margin-top: 30px; text-align: center; border-top: 1px solid #ccc; padding-top: 20px;">
        <p>📧 Developed by Rohit Kattimani and CodeSquad Team (<a href="https://github.com/RohitKattimani/">GitHub</a>)</p>
    </div>
    """)

app.launch(share=True)

