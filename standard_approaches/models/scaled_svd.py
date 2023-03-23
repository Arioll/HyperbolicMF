import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import svds

from ..utils.dataprep import generate_interactions_matrix
from ..utils.evaluation import downvote_seen_items, topn_recommendations, model_evaluate


def rescale_matrix(matrix, scaling_factor):
    frequencies = matrix.getnnz(axis=0)
    scaling_weights = np.power(frequencies, 0.5 * (scaling_factor - 1))
    return matrix.dot(diags(scaling_weights)), scaling_weights


def build_ssvd_model(config, data, data_description):
    source_matrix = generate_interactions_matrix(data, data_description, rebase_users=False)
    scaled_matrix, scaling_weights = rescale_matrix(source_matrix, config['scaling'])

    *_, vt = svds(scaled_matrix, k=config['rank'], return_singular_vectors='vh')
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors, scaling_weights


def ssvd_scoring(params, data, data_description):
    item_factors, scaling_weights = params

    test_matrix = generate_interactions_matrix(data, data_description, rebase_users=True)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    downvote_seen_items(scores, data, data_description)

    return scores


def ssvd_grid_search(ranks, scalings, training, testset, holdout, data_description, topn=20):
    max_rank = max(ranks)
    config = {'rank': max_rank}
    results = {}
    for scaling in scalings:
        config['scaling'] = scaling
        item_factors, scaling_weights = build_ssvd_model(config, training, data_description)
        for rank in ranks:
            item_factors_trunc = item_factors[:, :rank]
            scores = ssvd_scoring((item_factors_trunc, scaling_weights), testset, data_description)
            recs = topn_recommendations(scores, topn=topn)
            results[(rank, scaling)] = model_evaluate(recs, holdout, data_description)
    
    return results
