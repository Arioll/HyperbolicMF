import numpy as np
import scipy as sp
import cupy as cp
from tqdm.notebook import tqdm

class HyperbolicMF:

    def __init__(self, sigma_U=1e-2, sigma_V=1e-2):
        self.sigma_U = sigma_U
        self.sigma_V = sigma_V

    def predict(self, user_index, item_index):
        prod = self.compute_lorentzian_distance(user_index, item_index)
        exp_prod = np.exp(-prod)
        return exp_prod / (1 + exp_prod)

    def add_unknown_user(self, user_relations, weights, n_iters, lr):
        n_new_users, _ = user_relations.shape
        user_vectors = np.random.randn(user_relations.shape[0], self.decomposition_rank) * self.sigma_U
        # self.U = np.concatenate((self.U, user_vector))
        for epoch in range(n_iters):
            for i in range(n_new_users):
                self.update_U_row(user_vectors, user_relations, weights, i, lr)
        new_user_indices = np.arange(n_new_users) + self.U.shape[0]
        self.U = np.concatenate((self.U, user_vectors))
        return new_user_indices # Method returns indices assigned to the new users

    def add_unknown_item(self, item_relations, weights, n_iters, lr):
        _, n_new_items = item_relations.shape
        item_vectors = np.random.randn(item_relations.shape[1], self.decomposition_rank) * self.sigma_U
        # self.U = np.concatenate((self.U, user_vector))
        for epoch in range(n_iters):
            for i in range(n_new_items):
                self.update_V_row(item_vectors, item_relations, weights, i, lr)
        new_item_indices = np.arange(n_new_items) + self.V.shape[0]
        self.V = np.concatenate((self.V, item_vectors))
        return new_item_indices # Method returns indices assigned to the new items

    def project_gradient_onto_hyperbolic_space(self, grad, origin):
        return grad + (grad @ origin) * origin

    def update_U_row(self, U_matrix, relations, weights, row_ind, lr):
        n_items, d = self.V.shape
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        U_grad = np.zeros(d + 1)
        p = self.predict([row_ind], np.arange(n_items))
        if weights is not None:
            weighted_diff = weights[row_ind] * (p - relations[row_ind])
        else:
            weighted_diff = p - relations[row_ind]
        # print(weighted_diff.shape, self.V[:, :-1].shape, self.V[:, -1].shape)
        # raise Exception('A')
        U_grad[:-1] = 2 * weighted_diff @ self.V[:, :-1] #/ n_items
        u_d_plus = U_matrix[row_ind, -1]
        # U_grad[-1] = -(2 * weighted_diff @ self.V[:, -1] #/ n_items 
        #                       + 2 * a_U * np.arccosh(u_d_plus) / np.sqrt(u_d_plus ** 2 - 1) 
        #                       + (d - 1) * u_d_plus / (u_d_plus ** 2 - 1) 
        #                       - (d - 1) / np.sqrt(u_d_plus ** 2 - 1) / np.arccosh(u_d_plus))
        # print(U_matrix[row_ind, :-1], u_d_plus, U_grad[:-1])
        U_grad[-1] -= 2 * weighted_diff @ self.V[:, -1] / n_items 
        # print(U_grad[-1])
        U_grad[-1] -= 2 * a_U * np.arccosh(u_d_plus) / np.sqrt(u_d_plus ** 2 - 1) 
        # print(U_grad[-1])
        U_grad[-1] -= (d - 1) * u_d_plus / (u_d_plus ** 2 - 1) 
        # print(U_grad[-1])
        U_grad[-1] += (d - 1) / np.sqrt(u_d_plus ** 2 - 1) / np.arccosh(u_d_plus)
        # print(U_grad[-1])
        # raise Exception('A')
        grad = self.project_gradient_onto_hyperbolic_space(U_grad, U_matrix[row_ind])
        # print(grad)
        U_matrix[row_ind, :] = np.exp(-lr * grad)


    def update_V_row(self, V_matrix, relations, weights, row_ind, lr):
        n_users, d = self.U.shape
        d = self.decomposition_rank - 1
        a_V = 1 / (2 * self.sigma_V ** 2)
        V_grad = np.zeros(d + 1)
        p = self.predict(np.arange(n_users), [row_ind])
        if weights is not None:
            weighted_diff = weights[:, row_ind] * (p - relations[:, row_ind])
        else:
            weighted_diff = p - relations[:, row_ind]
        # print(weighted_diff.shape, self.U[:, :-1].shape, self.U[:, -1].shape)
        V_grad[:-1] = 2 * weighted_diff.T @ self.U[:, :-1]
        v_d_plus = V_matrix[row_ind, -1]
        V_grad[-1] = -(2 * weighted_diff.T @ self.U[:, -1] / n_users + 
                              2 * a_V * np.arccosh(v_d_plus) / np.sqrt(v_d_plus ** 2 - 1) + 
                              (d - 1) * v_d_plus / (v_d_plus ** 2 - 1) - 
                              (d - 1) / np.sqrt(v_d_plus ** 2 - 1) / np.arccosh(v_d_plus))
        grad = self.project_gradient_onto_hyperbolic_space(V_grad, V_matrix[row_ind])
        # print(grad)
        # raise Exception('A')
        V_matrix[row_ind, :] = np.exp(-lr * grad)

    def compute_loss(self, relations, weights, n_batches):
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        a_V = 1 / (2 * self.sigma_V ** 2)
        loss = 0
        U_batch_size = self.U.shape[0] // n_batches if self.U.shape[0] % n_batches == 0 else self.U.shape[0] // n_batches + 1
        V_batch_size = self.V.shape[0] // n_batches if self.V.shape[0] % n_batches == 0 else self.V.shape[0] // n_batches + 1
        for i in range(self.U.shape[0] // U_batch_size):
            U_inds = np.arange(U_batch_size * i, U_batch_size * (i + 1))
            for j in range(self.V.shape[0] // V_batch_size):
                V_inds = np.arange(V_batch_size * j, V_batch_size * (j + 1))
                d_L = self.compute_lorentzian_distance(U_inds, V_inds)
                rel = relations[U_batch_size * i:U_batch_size * (i + 1), V_batch_size * j:V_batch_size * (j + 1)]
                if weights is not None:
                    first_term = ((np.log(1 + np.exp(-d_L)) + rel.toarray() * d_L) * weights).sum()
                else:
                    first_term = (np.log(1 + np.exp(-d_L)) + rel.toarray() * d_L).sum()
                loss += first_term
        U_term = a_U * np.arccosh(self.U[:, -1])**2 + (d - 1) * np.log(np.sqrt(self.U[:, -1]**2 - 1) / np.arccosh(self.U[:, -1]))
        V_term = a_V * np.arccosh(self.V[:, -1])**2 + (d - 1) * np.log(np.sqrt(self.V[:, -1]**2 - 1) / np.arccosh(self.V[:, -1]))
        loss += U_term.sum() + V_term.sum()
        return loss

    def compute_lorentzian_distance(self, user_inds, item_inds):
        U = self.U[user_inds, :]
        V = self.V[item_inds, :]
        U[:, -1] = -U[:, -1]
        return -2 - 2 * U @ V.T

    def draw_from_pseudo_hyperbolic_gaussian(self, shape, sigma, return_gauss=False):
        x = np.random.randn(*shape) * sigma
        x[:, -1] = 0
        mu = np.array([0] * (shape[1] - 1) + [1])#np.zeros(shape[1])
        # mu[-1] = 1
        mask = np.ones(x.shape)
        mask[:, -1] = -1
        norm_y = np.sqrt((x * x * mask).sum(axis=1, keepdims=True))
        if not return_gauss:
            return np.cosh(norm_y) * mu + np.sinh(norm_y) * x / norm_y
        return np.cosh(norm_y) * mu + np.sinh(norm_y) * x / norm_y, x

    # HAGD algorithm from paper
    def fit(self, relation_matrix, weight_matrix, num_epochs, learning_rate=1e-3, decomposition_rank=5):
        n_users, n_items = relation_matrix.shape
        self.U = self.draw_from_pseudo_hyperbolic_gaussian((relation_matrix.shape[0], decomposition_rank), self.sigma_U)
        self.V = self.draw_from_pseudo_hyperbolic_gaussian((relation_matrix.shape[1], decomposition_rank), self.sigma_V)
        self.decomposition_rank = decomposition_rank
        losses = []
        iterator = tqdm(range(num_epochs))
        for epoch in iterator:
            for i in range(n_users):
                self.update_U_row(self.U, relation_matrix, weight_matrix, i, learning_rate)
            for j in range(n_items):
                self.update_V_row(self.V, relation_matrix, weight_matrix, j, learning_rate)
            loss = self.compute_loss(relation_matrix, weight_matrix, 1000)
            iterator.set_description(f'Epoch: {epoch + 1} \t Loss: {float(loss):.4f}')
            losses.append(float(loss))
        return losses