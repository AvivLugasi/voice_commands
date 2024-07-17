from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_sim(vec1, vec2, is_1d=False):
    if is_1d:
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]
