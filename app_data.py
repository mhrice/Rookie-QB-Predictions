from fastai.tabular.all import *
import gradio as gr
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
# path = Path()
df = pd.read_csv("rookie_year.csv")
learn = load_learner("export.pkl")

def predict2(data):
    row = data.drop("Name", axis=1).astype(float)
    row["Cmp"] = row["Att"].item() * row["Cmp%"].item()
    pred_row, clas, probs = learn.predict(row.iloc[0])
    prediction = pred_row.decode()["Tier"].item()    
    return prediction


demo2 = gr.Interface(fn=predict2, 
                    inputs=gr.Dataframe(row_count=1, col_count=8, headers=[x for x in columns if x not in ["Cmp", "G", "GS"]], label="Rookie Year Stats"), 
                    outputs=gr.Textbox(label="Prediction"),
                    title="Rookie QB Career Prediction (Stats)",
                    description="Given stats of a presumed rookie QB, predict their career tier. Uses data from https:\/\/www.pro-football-reference.com. Tiers based on PFR Approximate Value.",
                    article="See more details at https://github.com/mhrice/Rookie-QB-Predictions"
                    )

demo2.launch()