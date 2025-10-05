from catboost import CatBoostClassifier
import ast
import gradio as gr
import pandas as pd
import joblib
import os

"""Load the model and datasets"""

# Loading trained model and mlb(I have for ease just added the pickle files in modeldata folder, if you want to train it on your machine locally, then do it running model.py)
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

# Convert all strings to a list in the dictionary
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
df_combined['DrugBankID'] = df_combined['DrugBankID'].apply(convert_string_to_list)

# Drop unknown IDs from HODDI dataset (this list is constant)
unknown_ids = ['DB03862', 'DB04482', 'DB04920', 'DB11050', 'DB12366', 'DB13151', 'DB14693', 'DB15270', 'DB18046']

def has_unknown(drug_ids):
    return any(d in unknown_ids for d in drug_ids)

df_combined = df_combined[~df_combined['DrugBankID'].apply(has_unknown)].reset_index(drop=True)

# Create name-to-ID mapping
name_to_id = dict(zip(df_dict['Name'], df_dict['DrugBankID']))
id_to_name = dict(zip(df_dict['DrugBankID'], df_dict['Name']))

def name_to_id_mapping(drug_names):
    """Convert drug names to DrugBank IDs with error handling"""
    drug_ids = []
    missing_drugs = []

    for name in drug_names:
        # Remove extra whitespace by stripping and standardize case 
        clean_name = name.strip().lower()

        # Find matching drug (case insensitive)
        match = None
        for dict_name, drug_id in name_to_id.items():
            if clean_name == dict_name.strip().lower():
                match = drug_id
                break

        if match:
            drug_ids.append(match)
        else:
            missing_drugs.append(name)

    return drug_ids, missing_drugs

def extract_valid_drugs(interaction_df, drug_dict_df):
    """
    Extract DrugBank IDs and names that exist in both datasets
    """
    # Extract all unique DrugBank IDs from interaction dataset
    all_interaction_ids = set()
    for drug_list in interaction_df['DrugBankID']:
        all_interaction_ids.update(drug_list)

    # Filter drug dictionary to only include IDs present in interactions
    valid_drugs = drug_dict_df[drug_dict_df['DrugBankID'].isin(all_interaction_ids)]

    # Create mappings
    id_to_name = dict(zip(valid_drugs['DrugBankID'], valid_drugs['Name']))
    name_to_id = dict(zip(valid_drugs['Name'], valid_drugs['DrugBankID']))
    drug_name_list = valid_drugs['Name'].unique().tolist()

    return id_to_name, name_to_id, drug_name_list

# Get mappings and drug list
id_to_name, name_to_id, drug_list = extract_valid_drugs(df_combined, df_dict)

def prepare_model_input(drug_names):
    """Convert drug names to model-ready features"""
    try:
        drug_ids = [name_to_id[name] for name in drug_names]
        encoded = mlb.transform([drug_ids]).astype(float)
        return encoded
    except KeyError as e:
        print(f"Drug not found: {e}")
        return None
    except Exception as e:
        print(f"Input error: {e}")
        return None

def predict_combination(drugs):
    """Make prediction with selected drugs"""
    try:
        if len(drugs) < 2:
            return "Please select at least 2 drugs", "0%"

        # Prepare model input with DrugBank IDs
        input_data = prepare_model_input(drugs)
        if input_data is None:
            return "Error preparing input", "0%"

        # Get prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100

        result = "Risky" if prediction == 1 else "Safe"
        return result, f"{confidence:.0f}%"

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction error", "0%"

with gr.Blocks() as app:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>💊 MedSafe 💊</h1>
        <p>
            Welcome to the MedSafe Platfrom! <br>
            Taking multiple medications can sometimes lead to dangerous interactions, which are a major cause of preventable harm in healthcare. <br>
            Every year, thousands of serious adverse drug events occur, some of which can be life-threatening. <br>
            Checking all possible drug combinations manually is nearly impossible due to the sheer number of medications and hidden interaction patterns. <br>
            This tool uses advanced machine learning to quickly predict potential risks, helping healthcare professionals and patients make safer medication choices.
        </p>
        <h3>How does it work?</h3>
        <p>
            To use the predictor, select at least 2 drugs from the list, and click on Predict button<br>
            The model will provide a <strong>safety prediction</strong> for the chosen combination, along with a confidence score that shows how certain it is about the result. <br>
            This helps you quickly identify potentially risky drug interactions before they happen.
        </p>
    </div>
    """)


    with gr.Row():
        drug_select = gr.Dropdown(
            choices=drug_list,
            label="Select your Drug Names",
            multiselect=True,
            interactive=True
        )

    with gr.Row():
        selected_display = gr.Textbox(label="Selected Drugs", interactive=False)
        output = gr.Textbox(label="Prediction Result")
        confidence = gr.Textbox(label="Confidence Score")

    # State management
    selected_drugs = gr.State([])

    def update_selection(drugs):
        return ", ".join(drugs) if drugs else "", drugs

    drug_select.change(
        fn=update_selection,
        inputs=drug_select,
        outputs=[selected_display, selected_drugs]
    )

    submit_btn = gr.Button("Predict", variant="stop")
    submit_btn.click(
        fn=predict_combination,
        inputs=selected_drugs,
        outputs=[output, confidence]
    )

    clear_btn = gr.Button("Clear")
    clear_btn.click(
        fn=lambda: ("", [], [], "", ""),
        outputs=[selected_display, selected_drugs, drug_select, output, confidence]
    )

    # Contact box at the bottom
    gr.Markdown("""
    <div style="margin-top: 30px; text-align: center; border-top: 1px solid #ccc; padding-top: 20px;">
        <p>📧 This GRADIO platfrom is developed by:
        Rohit Kattimani (<a href="https://github.com/RohitKattimani/">RohitKattimani</a>)
        </p>
    </div>
    """)

app.launch(share=True)
