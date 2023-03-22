import numpy as np
import scipy as sp
import cupy as cp
from tqdm.notebook import tqdm


class HyperbolicMF:

    def __init__(self, sigma_U=1e-2, sigma_V=1e-2, backend=np):
        self.sigma_U = sigma_U
        self.sigma_V = sigma_V
        self.backend = backend
        self.U_grad_norm = 0
        self.V_grad_norm = 0
        self.op_C = 1e-10

    def predict(self, user_index, item_index):
        prod = self.compute_lorentzian_distance(user_index, item_index)
        # print(prod.shape)
        exp_prod = self.backend.exp(-prod)
        return exp_prod / (1 + exp_prod)

    def add_unknown_user(self, user_relations, weights, n_iters, lr, batch_size, stop_criterion=1e-4):
        n_new_users, _ = user_relations.shape
        user_vectors = self.draw_from_pseudo_hyperbolic_gaussian((user_relations.shape[0], self.decomposition_rank), self.sigma_U)
        U_n_batches = int(self.backend.ceil(n_new_users / batch_size))
        U_norms = []
        for epoch in range(n_iters):
            for i in range(U_n_batches):
                indexes = self.backend.arange(i * batch_size, min(n_new_users, (i+1) * batch_size))
                self.update_U_row(user_vectors, user_relations, weights, indexes, lr)
            U_norms.append(self.U_grad_norm)
            self.U_grad_norm = 0
            if U_norms[-1] < stop_criterion:
                break
        new_user_indices = self.backend.arange(n_new_users) + self.U.shape[0]
        self.U = self.backend.concatenate((self.U, user_vectors))
        return new_user_indices, U_norms # Method returns indices assigned to the new users

    def add_unknown_item(self, item_relations, weights, n_iters, lr, batch_size, stop_criterion=1e-4):
        _, n_new_items = item_relations.shape
        item_vectors = self.draw_from_pseudo_hyperbolic_gaussian((item_relations.shape[1], self.decomposition_rank), self.sigma_V)
        V_n_batches = int(self.backend.ceil(n_new_items / batch_size))
        V_norms = []
        for epoch in range(n_iters):
            for i in range(V_n_batches):
                indexes = self.backend.arange(i * batch_size, min(n_new_items, (i+1) * batch_size))
                self.update_V_row(item_vectors, item_relations, weights, indexes, lr)
            V_norms.append(self.V_grad_norm)
            self.V_grad_norm = 0
            if V_norms[-1] < stop_criterion:
                break
        new_item_indices = self.backend.arange(n_new_items) + self.V.shape[0]
        self.V = self.backend.concatenate((self.V, item_vectors))
        return new_item_indices, V_norms # Method returns indices assigned to the new items

    def project_gradient_onto_tangent_space(self, grad, origin):
        if len(grad.shape) == 1:
            grad = grad.reshape((1, grad.shape[0]))
        if len(origin.shape) == 1:
            origin = origin.reshape((1, origin.shape[0]))
        mask = self.backend.ones(origin.shape)
        mask[:, -1] = -1
        return grad + (grad * origin * mask).sum(axis=1, keepdims=True) * origin

    def update_U_row(self, U_matrix, relations, weights, row_ind, lr):
        n_items, d = self.V.shape
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        U_grad = self.backend.zeros((len(row_ind), d + 1))
        p = self.predict(row_ind, self.backend.arange(n_items))

        if weights is not None:
            w = weights[row_ind, :].toarray()
            w[w == 0] = 1
            weighted_diff = w * (p - relations[row_ind, :])
        else:
            weighted_diff = p - relations[row_ind, :]

        U_grad[:, :-1] = 2 * weighted_diff @ self.V[:, :-1]
        u_d_plus = self.backend.maximum(U_matrix[row_ind, -1], self.backend.ones_like(U_matrix[row_ind, -1]) + self.op_C)
        # U_grad[U_matrix[row_ind, -1] > 1, -1] = 2 * weighted_diff @ self.V[:, -1] - 
        U_grad[:, -1] += 2 * weighted_diff @ self.V[:, -1]
        U_grad[:, -1] -= 2 * a_U * self.backend.arccosh(u_d_plus) / (self.backend.sqrt(u_d_plus ** 2 - 1) + self.op_C)
        U_grad[:, -1] -= (d - 1) * (u_d_plus * self.backend.arccosh(u_d_plus) - self.backend.sqrt(u_d_plus ** 2 - 1)) / ((u_d_plus ** 2 - 1) * self.backend.arccosh(u_d_plus) + self.op_C)
        grad = self.project_gradient_onto_tangent_space(U_grad, U_matrix[row_ind, :])
        grad = self.exp_mu(U_matrix[row_ind, :], -lr * grad)
        self.U_grad_norm += self.backend.linalg.norm(U_matrix[row_ind, :] - grad).get()
        U_matrix[row_ind, :] = grad


    def update_V_row(self, V_matrix, relations, weights, row_ind, lr):
        # if not isinstance(row_ind, list):
        #     row_ind = [row_ind]
        n_users, d = self.U.shape
        d = self.decomposition_rank - 1
        a_V = 1 / (2 * self.sigma_V ** 2)
        V_grad = self.backend.zeros((len(row_ind), d + 1))
        p = self.predict(self.backend.arange(n_users), row_ind)

        if weights is not None:
            w = weights[:, row_ind].toarray()
            w[w == 0] = 1
            weighted_diff = w * (p - relations[:, row_ind])
        else:
            weighted_diff = p - relations[:, row_ind]

        V_grad[:, :-1] = 2 * weighted_diff.T @ self.U[:, :-1]
        v_d_plus = self.backend.maximum(V_matrix[row_ind, -1], self.backend.ones_like(V_matrix[row_ind, -1]) + self.op_C)
        V_grad[:, -1] += 2 * weighted_diff.T @ self.U[:, -1]
        V_grad[:, -1] -= 2 * a_V * self.backend.arccosh(v_d_plus) / (self.backend.sqrt(v_d_plus ** 2 - 1) + self.op_C)
        V_grad[:, -1] -= (d - 1) * (v_d_plus * self.backend.arccosh(v_d_plus) - self.backend.sqrt(v_d_plus ** 2 - 1)) / ((v_d_plus ** 2 - 1) * self.backend.arccosh(v_d_plus) + self.op_C)
        grad = self.project_gradient_onto_tangent_space(V_grad, V_matrix[row_ind, :])
        grad = self.exp_mu(V_matrix[row_ind, :], -lr * grad)
        self.V_grad_norm += self.backend.linalg.norm(V_matrix[row_ind, :] - grad).get()
        V_matrix[row_ind, :] = grad

    def compute_loss(self, relations, weights, batch_size):
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        a_V = 1 / (2 * self.sigma_V ** 2)
        loss = 0
        U_n_batches = int(self.backend.ceil(self.U.shape[0] / batch_size))#self.U.shape[0] // batch_size if self.U.shape[0] % batch_size == 0 else self.U.shape[0] // batch_size + 1
        V_n_batches = int(self.backend.ceil(self.V.shape[0] / batch_size))#self.V.shape[0] // batch_size if self.V.shape[0] % batch_size == 0 else self.V.shape[0] // batch_size + 1
        for i in range(U_n_batches):
            U_inds = self.backend.arange(batch_size * i, min(batch_size * (i + 1), self.U.shape[0]))
            for j in range(V_n_batches):
                # print(i, j)
                V_inds = self.backend.arange(batch_size * j, min(batch_size * (j + 1), self.V.shape[0]))
                d_L = self.compute_lorentzian_distance(U_inds, V_inds)
                rel = relations[batch_size * i:batch_size * (i + 1), batch_size * j:batch_size * (j + 1)]
                if weights is not None:
                    w = weights[batch_size * i:batch_size * (i + 1), batch_size * j:batch_size * (j + 1)].toarray()
                    w[w == 0] = 1
                    first_term = (w * (self.backend.log(1 + self.backend.exp(-d_L)) + rel.toarray() * d_L)).sum()
                else:
                    first_term = (self.backend.log(1 + self.backend.exp(-d_L)) + rel.toarray() * d_L).sum()
                loss += first_term
        
        U_last = self.backend.maximum(self.U[:, -1], self.backend.ones_like(self.U[:, -1]) + self.op_C) # To avoid cases when self.U[:, -1] < 1 due to numerical errors
        V_last = self.backend.maximum(self.V[:, -1], self.backend.ones_like(self.V[:, -1]) + self.op_C)
        U_last_arccosh = self.backend.arccosh(U_last)
        V_last_arccosh = self.backend.arccosh(V_last)
        U_term = a_U * U_last_arccosh**2 + (d - 1) * self.backend.log(self.backend.sqrt(U_last**2 - 1) / U_last_arccosh)
        V_term = a_V * V_last_arccosh**2 + (d - 1) * self.backend.log(self.backend.sqrt(V_last**2 - 1) / V_last_arccosh)
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
        mask = self.backend.ones(x.shape)
        mask[:, -1] = -1
        prod = (x * x * mask).sum(axis=1, keepdims=True)
        squared_norm = self.backend.maximum(prod, self.backend.zeros_like(prod))
        norm_x = self.backend.sqrt(squared_norm)
        return self.backend.cosh(norm_x) * mu + self.backend.sinh(norm_x) * x / (norm_x + self.op_C)

    def draw_from_pseudo_hyperbolic_gaussian(self, shape, sigma, return_gauss=False):
        x = self.backend.random.randn(*shape) * sigma
        x[:, -1] = 0
        mu = self.backend.array([0] * (shape[1] - 1) + [1])
        exp_mu_x = self.exp_mu(mu, x)
        if not return_gauss:
            return exp_mu_x
        return exp_mu_x, x

    # HAGD algorithm from paper
    def fit(self, relation_matrix, weight_matrix, num_epochs, learning_rate=1e-3, decomposition_rank=5, batch_size=10, seed=42, stop_criterion=1e-4):
        if seed is not None:
            self.backend.random.seed(seed)
        n_users, n_items = relation_matrix.shape
        self.U = self.draw_from_pseudo_hyperbolic_gaussian((n_users, decomposition_rank), self.sigma_U)
        self.V = self.draw_from_pseudo_hyperbolic_gaussian((n_items, decomposition_rank), self.sigma_V)
        U_n_batches = int(self.backend.ceil(n_users / batch_size))
        V_n_batches = int(self.backend.ceil(n_items / batch_size))
        self.decomposition_rank = decomposition_rank
        losses = []
        U_norms = []
        V_norms = []
        iterator = tqdm(range(num_epochs))
        for epoch in iterator:
            # if len(U_norms) > 1 and U_norms[-1] > 1e-3:
            for i in range(U_n_batches):
                indices = self.backend.arange(i * batch_size, min((i+1) * batch_size, n_users))
                self.update_U_row(self.U, relation_matrix, weight_matrix, indices, learning_rate)
            # if len(V_norms) > 1 and V_norms[-1] > 1e-3:
            for j in range(V_n_batches):
                indices = self.backend.arange(j * batch_size, min((j+1) * batch_size, n_items))
                self.update_V_row(self.V, relation_matrix, weight_matrix, indices, learning_rate)
            loss = self.compute_loss(relation_matrix, weight_matrix, 1000).get()
            iterator.set_description(f'Epoch: {epoch + 1} \t Loss: {float(loss):.4f} \t U grad norm: {float(self.U_grad_norm):.4f} \t V grad norm: {float(self.V_grad_norm):.4f}')
            U_norms.append(self.U_grad_norm)
            V_norms.append(self.V_grad_norm)
            self.U_grad_norm = 0
            self.V_grad_norm = 0
            if U_norms[-1] < stop_criterion and V_norms[-1] < stop_criterion:
                break
            losses.append(float(loss))
            assert not np.isnan(loss)
        return losses, U_norms, V_norms

    # def fit_samplewise(self, relation_matrix, weight_matrix, num_epochs, learning_rate=1e-3, decomposition_rank=5, batch_size=10, seed=42, stop_criterion=1e-4):
    #     if seed is not None:
    #         self.backend.random.seed(seed)
    #     n_users, n_items = relation_matrix.shape
    #     self.U = self.draw_from_pseudo_hyperbolic_gaussian((n_users, decomposition_rank), self.sigma_U)
    #     self.V = self.draw_from_pseudo_hyperbolic_gaussian((n_items, decomposition_rank), self.sigma_V)
    #     U_n_batches = int(self.backend.ceil(n_users / batch_size))
    #     V_n_batches = int(self.backend.ceil(n_items / batch_size))
    #     self.decomposition_rank = decomposition_rank
