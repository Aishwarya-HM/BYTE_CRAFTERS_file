import pickle
import sklearn 

# Replace 'your_file.pkl' with the path to your .pkl file
pkl_file_path = 'model.pkl'

# Load the .pkl file
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Print the deserialized data
print(data)
