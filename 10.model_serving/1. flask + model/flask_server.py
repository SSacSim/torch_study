import os
import json
import numpy as np
from flask import Flask, request

import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_model import ConvNet


############## model load
MODEL_WEIGHT_PATH = "./convnet.pth"
my_model = ConvNet()
my_model.load_state_dict(torch.load(MODEL_WEIGHT_PATH,map_location="cpu"))
my_model.eval()
##########################

def run_model(input_tensor):
    model_input = input_tensor.unsqueeze(0)
    with torch.no_grad():
        model_output = my_model(model_input)[0]
    model_prediction = model_output.detach().numpy().argmax()
    return model_prediction

def post_process(output):
    return str(output)


app = Flask(__name__)
@app.route("/test", methods=["POST"])

def test():
    data = request.files['data'].read()
    md = json.load(request.files['metadata'])
    input_array = np.frombuffer(data, dtype=np.float32)
    input_image_tensor = torch.from_numpy(input_array).view(md["dims"])
    output = run_model(input_image_tensor)
    final_output = post_process(output)
    return final_output 

if __name__ == "__main__":
    app.run(host= 'localhost', port = 8890)