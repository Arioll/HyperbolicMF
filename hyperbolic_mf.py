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

    def add_unknown_user(self, user_relations, weights, n_iters, lr, batch_size):
        n_new_users, _ = user_relations.shape
        user_vectors = self.draw_from_pseudo_hyperbolic_gaussian((user_relations.shape[0], self.decomposition_rank), self.sigma_U)
        U_n_batches = int(np.ceil(n_new_users / batch_size))
        for epoch in range(n_iters):
            for i in range(U_n_batches):
                indexes = np.arange(i * batch_size, min(n_new_users, (i+1) * batch_size))
                self.update_U_row(user_vectors, user_relations, weights, indexes, lr)
        new_user_indices = np.arange(n_new_users) + self.U.shape[0]
        self.U = np.concatenate((self.U, user_vectors))
        return new_user_indices # Method returns indices assigned to the new users

    def add_unknown_item(self, item_relations, weights, n_iters, lr, batch_size):
        _, n_new_items = item_relations.shape
        item_vectors = self.draw_from_pseudo_hyperbolic_gaussian((item_relations.shape[1], self.decomposition_rank), self.sigma_V)
        V_n_batches = int(np.ceil(n_new_items / batch_size))
        for epoch in range(n_iters):
            for i in range(V_n_batches):
                indexes = np.arange(i * batch_size, min(n_new_items, (i+1) * batch_size))
                self.update_V_row(item_vectors, item_relations, weights, indexes, lr)
        new_item_indices = np.arange(n_new_items) + self.V.shape[0]
        self.V = np.concatenate((self.V, item_vectors))
        return new_item_indices # Method returns indices assigned to the new items

    def project_gradient_onto_tangent_space(self, grad, origin):
        if len(grad.shape) == 1:
            grad = grad.reshape((1, grad.shape[0]))
        if len(origin.shape) == 1:
            origin = origin.reshape((1, origin.shape[0]))
        mask = np.ones(origin.shape)
        mask[:, -1] = -1
        return grad + (grad * origin * mask).sum(axis=1, keepdims=True) * origin

    def update_U_row(self, U_matrix, relations, weights, row_ind, lr):
        # if not isinstance(row_ind, list):
        #     row_ind = [row_ind]
        n_items, d = self.V.shape
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        U_grad = np.zeros((len(row_ind), d + 1))
        p = self.predict(row_ind, np.arange(n_items))
        if weights is not None:
            weighted_diff = weights[row_ind, :].multiply(p - relations[row_ind, :])
        else:
            weighted_diff = p - relations[row_ind, :]
        U_grad[:, :-1] = 2 * weighted_diff @ self.V[:, :-1]
        u_d_plus = U_matrix[row_ind, -1]
        U_grad[:, -1] += 2 * weighted_diff @ self.V[:, -1]
        U_grad[:, -1] -= 2 * a_U * np.arccosh(u_d_plus) / np.sqrt(u_d_plus ** 2 - 1)
        U_grad[:, -1] -= (d - 1) * u_d_plus / (u_d_plus ** 2 - 1)
        U_grad[:, -1] += (d - 1) / (np.sqrt(u_d_plus ** 2 - 1) * np.arccosh(u_d_plus))
        grad = self.project_gradient_onto_tangent_space(U_grad, U_matrix[row_ind, :])
        grad = self.exp_mu(U_matrix[row_ind, :], -lr * grad)
        U_matrix[row_ind, :] = grad


    def update_V_row(self, V_matrix, relations, weights, row_ind, lr):
        # if not isinstance(row_ind, list):
        #     row_ind = [row_ind]
        n_users, d = self.U.shape
        d = self.decomposition_rank - 1
        a_V = 1 / (2 * self.sigma_V ** 2)
        V_grad = np.zeros((len(row_ind), d + 1))
        p = self.predict(np.arange(n_users), row_ind)
        if weights is not None:
            weighted_diff = weights[:, row_ind].multiply(p - relations[:, row_ind])
        else:
            weighted_diff = p - relations[:, row_ind]
        V_grad[:, :-1] = 2 * weighted_diff.T @ self.U[:, :-1]
        v_d_plus = V_matrix[row_ind, -1]
        V_grad[:, -1] += 2 * weighted_diff.T @ self.U[:, -1]
        V_grad[:, -1] -= 2 * a_V * np.arccosh(v_d_plus) / np.sqrt(v_d_plus ** 2 - 1) 
        V_grad[:, -1] -= (d - 1) * v_d_plus / (v_d_plus ** 2 - 1) 
        V_grad[:, -1] += (d - 1) / (np.sqrt(v_d_plus ** 2 - 1) * np.arccosh(v_d_plus))
        grad = self.project_gradient_onto_tangent_space(V_grad, V_matrix[row_ind, :])
        grad = self.exp_mu(V_matrix[row_ind, :], -lr * grad)
        V_matrix[row_ind, :] = grad

    def compute_loss(self, relations, weights, batch_size):
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        a_V = 1 / (2 * self.sigma_V ** 2)
        loss = 0
        U_n_batches = int(np.ceil(self.U.shape[0] / batch_size))#self.U.shape[0] // batch_size if self.U.shape[0] % batch_size == 0 else self.U.shape[0] // batch_size + 1
        V_n_batches = int(np.ceil(self.V.shape[0] / batch_size))#self.V.shape[0] // batch_size if self.V.shape[0] % batch_size == 0 else self.V.shape[0] // batch_size + 1
        for i in range(U_n_batches):
            U_inds = np.arange(batch_size * i, min(batch_size * (i + 1), self.U.shape[0]))
            for j in range(V_n_batches):
                # print(i, j)
                V_inds = np.arange(batch_size * j, min(batch_size * (j + 1), self.V.shape[0]))
                d_L = self.compute_lorentzian_distance(U_inds, V_inds)
                rel = relations[batch_size * i:batch_size * (i + 1), batch_size * j:batch_size * (j + 1)]
                if weights is not None:
                    w = weights[batch_size * i:batch_size * (i + 1), batch_size * j:batch_size * (j + 1)]
                    first_term = (w.multiply(np.log(1 + np.exp(-d_L)) + rel.toarray() * d_L)).sum()
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

    def exp_mu(self, mu, x):
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        mask = np.ones(x.shape)
        mask[:, -1] = -1
        norm_x = np.sqrt((x * x * mask).sum(axis=1, keepdims=True))
        return np.cosh(norm_x) * mu + np.sinh(norm_x) * x / norm_x

    def draw_from_pseudo_hyperbolic_gaussian(self, shape, sigma, return_gauss=False):
        x = np.random.randn(*shape) * sigma
        x[:, -1] = 0
        mu = np.array([0] * (shape[1] - 1) + [1])
        exp_mu_x = self.exp_mu(mu, x)
        if not return_gauss:
            return exp_mu_x
        return exp_mu_x, x

    # HAGD algorithm from paper
    def fit(self, relation_matrix, weight_matrix, num_epochs, learning_rate=1e-3, decomposition_rank=5, batch_size=10):
        n_users, n_items = relation_matrix.shape
        self.U = self.draw_from_pseudo_hyperbolic_gaussian((n_users, decomposition_rank), self.sigma_U)
        self.V = self.draw_from_pseudo_hyperbolic_gaussian((n_items, decomposition_rank), self.sigma_V)
        U_n_batches = int(np.ceil(n_users / batch_size))#n_users // batch_size if n_users % batch_size == 0 else n_users // batch_size + 1
        V_n_batches = int(np.ceil(n_items / batch_size))#n_items // batch_size if n_items % batch_size == 0 else n_items // batch_size + 1
        self.decomposition_rank = decomposition_rank
        losses = []
        iterator = tqdm(range(num_epochs))
        for epoch in iterator:
            for i in range(U_n_batches):
                indices = np.arange(i * batch_size, min((i+1) * batch_size, n_users))
                self.update_U_row(self.U, relation_matrix, weight_matrix, indices, learning_rate)
            for j in range(V_n_batches):
                indices = np.arange(j * batch_size, min((j+1) * batch_size, n_items))
                self.update_V_row(self.V, relation_matrix, weight_matrix, indices, learning_rate)
            loss = self.compute_loss(relation_matrix, weight_matrix, 1000)
            iterator.set_description(f'Epoch: {epoch + 1} \t Loss: {float(loss):.4f}')
            losses.append(float(loss))
        return losses