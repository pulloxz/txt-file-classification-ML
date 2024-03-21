
import numpy as np
from scipy.sparse import csr_matrix


def trim_or_pad_features(matrix, expected_features):
    current_features = matrix.shape[1]

    if current_features < expected_features:
        padding = np.zeros((matrix.shape[0], expected_features - current_features))
        matrix = csr_matrix(np.hstack([matrix.toarray(), padding]))

    elif current_features > expected_features:
        matrix = matrix[:, :expected_features]

    return matrix
