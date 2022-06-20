from fastai.tabular.all import *
import gradio as gr
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
# path = Path()
df = pd.read_csv("rookie_year.csv")
learn = load_learner("export.pkl")
columns = ["Name", "G", "GS", "Cmp", "Att", "Yds", "Cmp%", "TD", "Int", "Y/G", "Sk"]

def predict(data):
    row = df[df["Name"] == data]
    row = row.loc[:, ~df.columns.str.contains('^Unnamed')]
    if not len(row):
        print("ERROR: No QB in database with this name")
        return        
    pred_row, clas, probs = learn.predict(row.iloc[0])
    prediction = pred_row.decode()["Tier"].item()    
    return row[columns], prediction

demo = gr.Interface(fn=predict, 
                    inputs=gr.Textbox(label="QB Name"), 
                    outputs=[
                        gr.Dataframe(row_count=1, col_count=11, headers=columns, label="Rookie Year Stats"), 
                        gr.Textbox(label="Prediction")
                    ],
                    title="Rookie QB Career Prediction (Name)",
                    description="Given Name of QB who has played in the NFL, predict their career tier. Uses data from https:\/\/www.pro-football-reference.com. Tiers based on PFR Approximate Value.",
                    article="See more details at https://github.com/mhrice/Rookie-QB-Predictions",
                    examples=["Tom Brady", "Joe Burrow", "Trevor Lawrence"]
                   )

demo.launch()