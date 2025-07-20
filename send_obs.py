import requests
# from envlogger import reader
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from pprint import pprint as pp
import base64

# Select a random episode from the TFDS dataset in the datasets folder in 'pegasus_drone_sim'
import tensorflow_datasets as tfds

ds = tfds.builder_from_directory('/home/ubuntu/datasets/pegasus_drone_sim/0.0.1')
sample = ds.as_dataset(split='train')
# Convert a small sample to DataFrame for easy viewing
# Note: This requires with_info=True if you're using tfds.load
sample_df = tfds.as_dataframe(sample.take(8), ds.info)
print(sample_df)
s1steps = sample_df.iloc[7]['steps']#[100:105]

nrows2 = []
for ind, s in enumerate(s1steps):
    if ind == 100:
        img = s['observation']['camera_image']
        break


data = {"full_image": img.tolist(), 'state': [0.12,0.012,0.3,0.1,0], 'instruction': 'attack the tank'}
response = requests.post('http://localhost:8777/act', json=data)

# Decode the base64 numpy arrays in the response
response_data = response.json()
print("Raw response structure:")
pp(response_data)

print("\nDecoded numpy arrays:")
for i, item in enumerate(response_data):
    if '__numpy__' in item:
        # Decode base64 string to bytes
        encoded_data = item['__numpy__']
        decoded_bytes = base64.b64decode(encoded_data)
        
        # Convert bytes back to numpy array
        dtype = np.dtype(item['dtype'])
        shape = tuple(item['shape'])
        array = np.frombuffer(decoded_bytes, dtype=dtype).reshape(shape)
        
        print(f"Array {i}: {array}")
        print(f"  Shape: {array.shape}, Dtype: {array.dtype}")
        print()