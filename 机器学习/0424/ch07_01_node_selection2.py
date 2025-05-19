def entropy(data, target):
    # Count occurrences of each target value
    target_counts = Counter(data[target])
    # Calculate total number of samples
    total_samples = len(data)
    # Calculate entropy
    entropy_value = 0
    for count in target_counts.values():
        probability = count / total_samples
        entropy_value -= probability * log2(probability)
    return entropy_value

def information_gain(data, feature, target):
    # Calculate entropy of the original dataset
    total_entropy = entropy(data, target)
    # Calculate weighted entropy after split
    feature_values = data[feature].unique()
    total_samples = len(data)
    weighted_entropy = 0
    
    for value in feature_values:
        subset = data[data[feature] == value]
        subset_samples = len(subset)
        weight = subset_samples / total_samples
        weighted_entropy += weight * entropy(subset, target)
    
    # Information gain is the reduction in entropy
    return total_entropy - weighted_entropy

def gini_index(data, target):
    # Count occurrences of each target value
    target_counts = Counter(data[target])
    # Calculate total number of samples
    total_samples = len(data)
    # Calculate Gini index
    gini = 1
    for count in target_counts.values():
        probability = count / total_samples
        gini -= probability ** 2
    return gini

def gini_gain(data, feature, target):
    # Calculate Gini index of the original dataset
    total_gini = gini_index(data, target)
    # Calculate weighted Gini index after split
    feature_values = data[feature].unique()
    total_samples = len(data)
    weighted_gini = 0
    
    for value in feature_values:
        subset = data[data[feature] == value]
        subset_samples = len(subset)
        weight = subset_samples / total_samples
        weighted_gini += weight * gini_index(subset, target)
    
    # Gini gain is the reduction in Gini index
    return total_gini - weighted_gini

def information_gain_ratio(data, feature, target):
    # Calculate information gain
    gain = information_gain(data, feature, target)
    
    # Calculate entropy of the feature (split information)
    feature_counts = Counter(data[feature])
    total_samples = len(data)
    split_info = 0
    for count in feature_counts.values():
        probability = count / total_samples
        split_info -= probability * log2(probability)
    
    # Avoid division by zero
    if split_info == 0:
        return 0
    
    # Calculate gain ratio
    return gain / split_info