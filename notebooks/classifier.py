import os
import json
import ndjson
import pandas as pd
import numpy as np

data_dir = "../data"
# categories = ["ant", "bear", "cat", "dog", "elephant"]  # Add or modify the list to include the animal categories you've downloaded
categories = ["ant"] 

data = []

for category in categories:
    file_path = os.path.join(data_dir, f"full_simplified_{category}.ndjson")
    with open(file_path) as f:
        drawings = ndjson.load(f)
        for drawing in drawings:
            data.append({"category": category, "drawing": drawing["drawing"]})

data_df = pd.DataFrame(data)