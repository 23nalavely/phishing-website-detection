import gradio as gr
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("best_phishing_model.pkl", "rb"))
features = pickle.load(open("feature_names.pkl", "rb"))
defaults = pickle.load(open("feature_defaults.pkl", "rb"))

def predict_phishing(length_url, length_hostname, nb_dots, nb_hyphens, page_rank):
    input_data = np.zeros(len(features))

    feature_map = {
        "length_url": length_url,
        "length_hostname": length_hostname,
        "nb_dots": nb_dots,
        "nb_hyphens": nb_hyphens,
        "page_rank": page_rank
    }

    for i, feature in enumerate(features):
        if feature in feature_map:
            input_data[i] = feature_map[feature]
        else:
            input_data[i] = defaults.get(feature, 0)

    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        return "Phishing Website"
    else:
        return "Legitimate Website"

app = gr.Interface(
    fn=predict_phishing,
    inputs=[
        gr.Number(label="Length URL"),
        gr.Number(label="Length Hostname"),
        gr.Number(label="Number of Dots"),
        gr.Number(label="Number of Hyphens"),
        gr.Number(label="Page Rank")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Phishing Website Detection",
    description="Aplikasi ini memprediksi apakah website termasuk legitimate atau phishing berdasarkan beberapa fitur URL."
)

app.launch()