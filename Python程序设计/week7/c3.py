import numpy as np

scores = []
groups = {'Excellent':0, 'Brilliant':0, 'Very Good':0, 'Good':0}
for score in scores:
    if score >= 90:
        groups['Excellent']+1;