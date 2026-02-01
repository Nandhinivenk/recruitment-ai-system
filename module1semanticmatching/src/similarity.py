from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(jd_embedding, resume_embedding):
    return cosine_similarity(jd_embedding, resume_embedding)[0][0]
