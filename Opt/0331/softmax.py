import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
     upstairs = np.exp(x)
     downstairs = upstairs.sum()
     fraction = upstairs / downstairs
     return fraction

x = [0.1,0.2,0.3,0.4,0.5]

print(softmax(x))