def standardize(tX):
    mean = np.mean(tX, axis=0)
    std = np.std(tX, axis=0)
    tX = (tX - mean) / std
    return tX

def add_bias(tX):
    return np.hstack((np.ones((tX.shape[0], 1)), tX))

def compute_accuracy(y, tX, w):
    mapping = np.vectorize((lambda x: -1 if 0 <= x < 0.5 else 1))
    predictions = mapping(sigmoid(tX @ w))
    return np.sum(y == predictions)/(y.shape[0])