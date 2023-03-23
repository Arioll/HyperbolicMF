import numpy as np
from scipy.sparse.linalg import svds

from src.utils.dataprep import generate_interactions_matrix
from src.utils.evaluation import downvote_seen_items, topn_recommendations, model_evaluate


def svd_scoring(params, data, data_description):
    item_factors = params

    test_matrix = generate_interactions_matrix(data, data_description, rebase_users=True)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    downvote_seen_items(scores, data, data_description)

    return scores


def build_svd_model(config, data, data_description):
    source_matrix = generate_interactions_matrix(data, data_description, rebase_users=False)
    *_, vt = svds(source_matrix, k=config['rank'], return_singular_vectors='vh')
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors


def svd_grid_search(ranks, training, testset, holdout, data_description, topn=20):
    max_rank = max(ranks)
    config = {'rank': max_rank}

    item_factors = build_svd_model(config, training, data_description)
    results = {}
    
    for rank in ranks:
        item_factors_trunc = item_factors[:, :rank]
        scores = svd_scoring(item_factors_trunc, testset, data_description)
        recs = topn_recommendations(scores, topn=topn)
        results[rank] = model_evaluate(recs, holdout, data_description)
    
    return results
