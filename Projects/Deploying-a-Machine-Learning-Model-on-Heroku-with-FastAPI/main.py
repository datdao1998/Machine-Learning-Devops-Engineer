# Put the code for your API here.

import os
import yaml
import uvicorn
from fastapi import FastAPI
from features import Features
import pandas as pd
from starter.inference_model import infer

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    # if os.system("dvc pull -f") != 0:
    #     exit("dvc pull failed")
    # os.system("rm -r .dvc .apt/usr/lib/dvc")
    # os.system("rm -r .dvc")

app = FastAPI()

with open('config.yml') as f:
    config = yaml.load(f, Loader= yaml.FullLoader)

@app.get("/")
async def root():
    return {"Message": "Hello!"}
        
@app.post("/inference")
async def inference(input_data: Features):
    input_data = input_data.dict()
    change_keys = config['normalize_keys']
    columns = config['columns']
    cat_features = config['cat_features']

    for normalized_key, key in change_keys:
        input_data[normalized_key] = input_data.pop(key)
    input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys()).T
    input_df = input_df[columns]

    prediction = infer(input_df, cat_features)

    return {"prediction": prediction}
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)