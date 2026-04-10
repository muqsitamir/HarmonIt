from sklearn.metrics.pairwise import cosine_similarity

def feature_similarity(feat1, feat2):
    return cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0][0]