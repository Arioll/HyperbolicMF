import numpy as np
import scipy as sp
import cupy as cp
from tqdm.notebook import tqdm
from movie_utils import generate_interactions_matrix
from random import randrange
from numba import njit


class HyperbolicMF:

    def __init__(self, sigma_U=1e-2, sigma_V=1e-2, backend=np):
        self.sigma_U = sigma_U
        self.sigma_V = sigma_V
        self.backend = backend
        self.U_grad_norm = 0
        self.V_grad_norm = 0
        self.op_C = 1e-10

    def flush_grad_norms(self):
        self.U_grad_norm = 0
        self.V_grad_norm = 0

    def predict(self, user_index, item_index):
        prod = self.compute_lorentzian_distance(user_index, item_index)
        exp_prod = self.backend.exp(-prod)
        # print(exp_prod.min())
        # print(self.backend.isnan(exp_prod).sum())
        return exp_prod / (1 + exp_prod)

    def add_unknown_user(self, user_relations, weights, n_iters, lr, batch_size, stop_criterion=1e-4):
        n_new_users, _ = user_relations.shape
        user_vectors = self.draw_from_pseudo_hyperbolic_gaussian((n_new_users, self.decomposition_rank), self.sigma_U)
        U_n_batches = int(self.backend.ceil(n_new_users / batch_size))
        U_norms = []
        for epoch in range(n_iters):
            for i in range(U_n_batches):
                indexes = self.backend.arange(i * batch_size, min(n_new_users, (i+1) * batch_size))
                self.update_U_row(user_vectors, user_relations, weights, indexes, -1, lr)
                # print('Pass')
            U_norms.append(self.U_grad_norm)
            self.flush_grad_norms()
            if U_norms[-1] < stop_criterion:
                break
        new_user_indices = self.backend.arange(n_new_users) + self.U.shape[0]
        self.U = self.backend.concatenate((self.U, user_vectors))
        return new_user_indices, U_norms # Method returns indices assigned to the new users

    def add_unknown_item(self, item_relations, weights, n_iters, lr, batch_size, stop_criterion=1e-4):
        _, n_new_items = item_relations.shape
        item_vectors = self.draw_from_pseudo_hyperbolic_gaussian((n_new_items, self.decomposition_rank), self.sigma_V)
        V_n_batches = int(self.backend.ceil(n_new_items / batch_size))
        V_norms = []
        for epoch in range(n_iters):
            for i in range(V_n_batches):
                indexes = self.backend.arange(i * batch_size, min(n_new_items, (i+1) * batch_size))
                self.update_V_row(item_vectors, item_relations, weights, indexes, -1, lr)
            V_norms.append(self.V_grad_norm)
            self.flush_grad_norms()
            if V_norms[-1] < stop_criterion:
                break
        new_item_indices = self.backend.arange(n_new_items) + self.V.shape[0]
        self.V = self.backend.concatenate((self.V, item_vectors))
        return new_item_indices, V_norms # Method returns indices assigned to the new items

    def add_unknown_user_stochastic(self, sampler, n_iters, lr, batch_size, stop_criterion=1e-4):
        n_new_users = sampler.n_users
        user_vectors = self.draw_from_pseudo_hyperbolic_gaussian((n_new_users, self.decomposition_rank), self.sigma_U)
        # U_n_batches = int(self.backend.ceil(n_new_users / batch_size))
        U_norms = []
        for epoch in range(n_iters):
            user_inds, item_inds = sampler.sample(batch_size)
            self.update_U_row(user_vectors, sampler.interactions, sampler.data, user_inds, item_inds, lr)
            U_norms.append(self.U_grad_norm)
            self.flush_grad_norms()
            # if U_norms[-1] < stop_criterion:
            #     break
        new_user_indices = self.backend.arange(n_new_users) + self.U.shape[0]
        self.U = self.backend.concatenate((self.U, user_vectors))
        return new_user_indices, U_norms # Method returns indices assigned to the new users

    def add_unknown_item_stochastic(self, sampler, n_iters, lr, batch_size, stop_criterion=1e-4):
        n_new_items = sampler.n_items
        item_vectors = self.draw_from_pseudo_hyperbolic_gaussian((n_new_items, self.decomposition_rank), self.sigma_V)
        # V_n_batches = int(self.backend.ceil(n_new_items / batch_size))
        V_norms = []
        for epoch in range(n_iters):
            user_inds, item_inds = sampler.sample(batch_size)
            self.update_V_row(item_vectors, sampler.interactions, sampler.data, item_inds, user_inds, lr)
            V_norms.append(self.V_grad_norm)
            self.flush_grad_norms()
            # if V_norms[-1] < stop_criterion:
            #     break
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
        # print('\t', (grad * origin * mask).sum(axis=1, keepdims=True))
        # print('\t', self.backend.linalg.norm(origin))
        # print('\t', self.backend.linalg.norm(grad))
        return grad + (grad * origin * mask).sum(axis=1, keepdims=True) * origin

    def update_U_row(self, U_matrix, relations, weights, row_ind, col_ind, lr):
        n_items, d = self.V.shape
        all_cols = False
        if isinstance(col_ind, int) and col_ind == -1:
            col_ind = self.backend.arange(n_items)
            all_cols = True
        d = self.decomposition_rank - 1
        a_U = 1 / (2 * self.sigma_U ** 2)
        # U_grad = self.backend.zeros((len(row_ind), d + 1))
        p = self.predict(row_ind, col_ind)
        assert not self.backend.any(self.backend.isnan(p))

        if weights is not None:
            if all_cols:
                w = weights[row_ind, :].toarray()
                w[w == 0] = 1
                weighted_diff = w * (p - relations[row_ind, :])
            else:
                w = weights[row_ind, col_ind]#.toarray()
                w[w == 0] = 1
                weighted_diff = w * (p - relations[row_ind, col_ind])
        else:
            if all_cols:
                weighted_diff = p - relations[row_ind, :]
            else:
                weighted_diff = p - relations[row_ind, col_ind]

        # U_grad[:, :-1] = 2 * weighted_diff @ self.V[col_ind, :-1]
        assert not self.backend.any(self.backend.isnan(self.V[col_ind, :]))
        assert not self.backend.any(self.backend.isnan(weighted_diff))
        U_grad = 2 * weighted_diff @ self.V[col_ind, :]# + 2 * U_matrix[row_ind, :] # L2 regularization
        assert not self.backend.any(self.backend.isnan(U_grad))
        assert not self.backend.any(self.backend.isinf(U_grad))
        try:
            assert not self.backend.any(self.backend.isnan(U_matrix[row_ind, :]))
            assert not self.backend.any(self.backend.isinf(U_matrix[row_ind, :]))
        except:
            print(row_ind)
            print(U_matrix[row_ind, -1])
            raise Exception()

        u_d_plus = self.backend.maximum(U_matrix[row_ind, -1], self.backend.ones_like(U_matrix[row_ind, -1]) + self.op_C)
        
        U_grad[:, -1] -= 2 * a_U * self.backend.arccosh(u_d_plus) / (self.backend.sqrt(u_d_plus ** 2 - 1) + self.op_C)

        try:
            assert not self.backend.any(self.backend.isnan(u_d_plus))
            assert not self.backend.any(self.backend.isinf(u_d_plus))
            assert not self.backend.any(self.backend.isnan(U_grad))
            assert not self.backend.any(self.backend.isinf(U_grad))
        except:
            print(u_d_plus.min(), u_d_plus.max())
            raise Exception()

        U_grad[:, -1] -= (d - 1) * (u_d_plus * self.backend.arccosh(u_d_plus) - self.backend.sqrt(u_d_plus ** 2 - 1)) / ((u_d_plus ** 2 - 1) * self.backend.arccosh(u_d_plus) + self.op_C)

        try:
            assert not self.backend.any(self.backend.isnan(U_grad))
            assert not self.backend.any(self.backend.isinf(U_grad))
        except:
            print(u_d_plus.min(), u_d_plus.max())
            raise Exception()

        U_grad = self.project_gradient_onto_tangent_space(U_grad, U_matrix[row_ind, :])
        assert not self.backend.any(self.backend.isnan(U_grad))
        assert not self.backend.any(self.backend.isinf(U_grad))

        U_grad = self.exp_mu(U_matrix[row_ind, :], -lr * U_grad)
        assert not self.backend.any(self.backend.isnan(U_grad))
        assert not self.backend.any(self.backend.isinf(U_grad))

        self.U_grad_norm += self.backend.linalg.norm(U_matrix[row_ind, :] - U_grad).get()
        U_matrix[row_ind, :] = U_grad


    def update_V_row(self, V_matrix, relations, weights, row_ind, col_ind, lr):
        n_users, d = self.U.shape
        all_cols = False
        if isinstance(col_ind, int) and col_ind == -1:
            col_ind = self.backend.arange(n_users)
            all_cols = True
        d = self.decomposition_rank - 1
        a_V = 1 / (2 * self.sigma_V ** 2)
        p = self.predict(col_ind, row_ind)

        if weights is not None:
            if all_cols:
                w = weights[:, row_ind].toarray()
                w[w == 0] = 1
                weighted_diff = w * (p - relations[:, row_ind])
            else:
                w = weights[col_ind, row_ind]#.toarray()
                w[w == 0] = 1
                weighted_diff = w * (p - relations[col_ind, row_ind])
        else:
            if all_cols:
                weighted_diff = p - relations[:, row_ind]
            else:
                weighted_diff = p - relations[col_ind, row_ind]

        assert not self.backend.any(self.backend.isnan(self.U[col_ind, :]))
        V_grad = 2 * weighted_diff.T @ self.U[col_ind, :]# + 2 * V_matrix[row_ind, :] # L2 regularization
        assert not self.backend.any(self.backend.isnan(V_grad))
        assert not self.backend.any(self.backend.isinf(V_grad))
        assert not self.backend.any(self.backend.isnan(V_matrix[row_ind, :]))
        assert not self.backend.any(self.backend.isinf(V_matrix[row_ind, :]))
        # print(self.backend.linalg.norm(V_grad))
        v_d_plus = self.backend.maximum(V_matrix[row_ind, -1], self.backend.ones_like(V_matrix[row_ind, -1]) + self.op_C)

        V_grad[:, -1] -= 2 * a_V * self.backend.arccosh(v_d_plus) / (self.backend.sqrt(v_d_plus ** 2 - 1) + self.op_C)
        try:
            assert not self.backend.any(self.backend.isnan(V_grad))
            assert not self.backend.any(self.backend.isinf(V_grad))
        except:
            print(v_d_plus.min(), v_d_plus.max())
            raise Exception()
        V_grad[:, -1] -= (d - 1) * (v_d_plus * self.backend.arccosh(v_d_plus) - self.backend.sqrt(v_d_plus ** 2 - 1)) / ((v_d_plus ** 2 - 1) * self.backend.arccosh(v_d_plus) + self.op_C)
        assert not self.backend.any(self.backend.isnan(V_grad))
        assert not self.backend.any(self.backend.isinf(V_grad))

        V_grad = self.project_gradient_onto_tangent_space(V_grad, V_matrix[row_ind, :])
        assert not self.backend.any(self.backend.isnan(V_grad))
        assert not self.backend.any(self.backend.isinf(V_grad))
        # print(self.backend.linalg.norm(V_grad[:, :-1]), self.backend.linalg.norm(V_grad[:, -1]))
        # print(self.backend.linalg.norm(V_matrix[row_ind, :]))
        V_grad = self.exp_mu(V_matrix[row_ind, :], -lr * V_grad)
        assert not self.backend.any(self.backend.isnan(V_grad))
        assert not self.backend.any(self.backend.isinf(V_grad))
        self.V_grad_norm += self.backend.linalg.norm(V_matrix[row_ind, :] - V_grad).get()
        V_matrix[row_ind, :] = V_grad

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
        # print(loss, U_term.sum(), V_term.sum())
        # loss += U_term.sum() + V_term.sum()
        return loss, U_term.sum(), V_term.sum()

    def compute_lorentzian_distance(self, user_inds, item_inds):
        U = self.U[user_inds, :]
        V = self.V[item_inds, :]
        U[:, -1] = -U[:, -1]
        return -2 - 2 * U @ V.T

    def exp_mu(self, mu, x):
        assert np.isinf(x).sum() == 0
        assert np.isnan(x).sum() == 0
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        mask = self.backend.ones(x.shape)
        mask[:, -1] = -1
        prod = (x * x * mask).sum(axis=1, keepdims=True)
        try:
            assert np.isinf(prod).sum() == 0
            assert np.isnan(prod).sum() == 0
        except:
            print(x.shape)
            print(prod)
            raise Exception()
        squared_norm = self.backend.maximum(prod, self.backend.zeros_like(prod))
        norm_x = self.backend.sqrt(squared_norm)
        try:
            assert np.isinf(norm_x).sum() == 0
            assert np.isnan(norm_x).sum() == 0
        except:
            print(x.shape)
            print(norm_x)
            raise Exception()
        # print(norm_x)
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
                self.update_U_row(self.U, relation_matrix, weight_matrix, indices, -1, learning_rate)
            # if len(V_norms) > 1 and V_norms[-1] > 1e-3:
            for j in range(V_n_batches):
                indices = self.backend.arange(j * batch_size, min((j+1) * batch_size, n_items))
                self.update_V_row(self.V, relation_matrix, weight_matrix, indices, -1, learning_rate)
            loss, U_t, V_t = self.compute_loss(relation_matrix, weight_matrix, 1000)#.get()
            iterator.set_description(f'Epoch: {epoch + 1} \t Loss: {float(loss):.4f} \t U term: {float(U_t):.4f} \t V term: {float(V_t):.4f}')
            U_norms.append(self.U_grad_norm)
            V_norms.append(self.V_grad_norm)
            self.flush_grad_norms()
            # if U_norms[-1] < stop_criterion and V_norms[-1] < stop_criterion:
            #     break
            losses.append(float(loss + U_t + V_t))
            assert not np.isnan(loss)
        return losses, U_norms, V_norms

    def fit_stochastic(self, sampler, num_iters, learning_rate=1e-3, decomposition_rank=5, batch_size=10, seed=42, use_weights=True, stop_criterion=1e-4):
        if seed is not None:
            self.backend.random.seed(seed)
        n_users, n_items = sampler.n_users, sampler.n_items
        self.U = self.draw_from_pseudo_hyperbolic_gaussian((n_users, decomposition_rank), self.sigma_U)
        self.V = self.draw_from_pseudo_hyperbolic_gaussian((n_items, decomposition_rank), self.sigma_V)
        self.decomposition_rank = decomposition_rank
        losses = []
        U_norms = []
        V_norms = []
        iterator = tqdm(range(num_iters))
        for it in iterator:
            users, items = sampler.sample(batch_size)
            inter, weights = sampler.interactions, sampler.data
            if not use_weights:
                weights = None
            self.update_U_row(self.U, inter, weights, users, items, learning_rate)
            self.update_V_row(self.V, inter, weights, items, users, learning_rate)
            if it % 100 == 0:
                loss, U_term, V_term = self.compute_loss(sampler.interactions, sampler.data, 5000)#.get()
                iterator.set_description(f'Iteration: {it + 1} \t Loss: {float(loss):.4f} \t U term: {float(U_term):.4f} \t V term: {float(V_term):.4f}')
            # iterator.set_description(f'Iteration: {it + 1} \t U grad norm: {float(self.U_grad_norm):.4f} \t V grad norm: {float(self.V_grad_norm):.4f}')
            U_norms.append(self.U_grad_norm)
            V_norms.append(self.V_grad_norm)
            self.flush_grad_norms()
            # if it > 0 and np.abs(loss - losses[-1]) < stop_criterion:
            #     break
            # losses.append(float(loss))
            # assert not np.isnan(loss)
        return losses, U_norms, V_norms

class NegativeSampler:

    def __init__(self, train_data, data_description, rebase_users, neg_samples_multiplier=2, backend=cp):
        self.interactions, self.data = generate_interactions_matrix(train_data, data_description, rebase_users, backend=backend)
        self.n_users, self.n_items = self.data.shape
        self.neg_samples_multiplier = neg_samples_multiplier
        self.backend = backend

        if backend == cp:
            self.non_zero_rows, self.non_zero_columns = self.data.get().nonzero()
            self.non_zero_rows, self.non_zero_columns = self.backend.array(self.non_zero_rows), self.backend.array(self.non_zero_columns)
        else:
            self.non_zero_rows, self.non_zero_columns = self.data.nonzero()
        
        self.negative_cols = self.sample_element_wise(self.interactions.indptr.get(), self.interactions.indices.get(), int(self.n_items), int(np.ceil(neg_samples_multiplier)))
        self.negative_users = self.backend.broadcast_to(
            self.backend.repeat(
                self.backend.arange(self.n_users),
                list(self.backend.diff(self.interactions.indptr).get())
            )[:, self.backend.newaxis],
            self.negative_cols.shape
        )
        self.negative_cols, self.negative_users = self.backend.array(self.negative_cols).reshape(-1), self.backend.array(self.negative_users).reshape(-1)

        self.n_nonzeros = len(self.non_zero_rows)
        self.n_zeros = len(self.negative_cols)

    def sample(self, batch_size):
        n_pos_samples = int(batch_size // (1 + self.neg_samples_multiplier))
        n_neg_samples = batch_size - n_pos_samples
        pos_inds = self.backend.random.choice(self.n_nonzeros, n_pos_samples, replace=False)
        neg_inds = self.backend.random.choice(self.n_zeros, n_neg_samples, replace=False)
        pos_users, pos_cols = self.non_zero_rows[pos_inds], self.non_zero_columns[pos_inds]
        neg_users, neg_cols = self.negative_users[neg_inds], self.negative_cols[neg_inds]
        # print(pos_users.shape, neg_users.shape)
        # print(pos_cols.shape, neg_cols.shape)
        all_users = self.backend.concatenate((pos_users, neg_users), axis=0)
        all_items = self.backend.concatenate((pos_cols, neg_cols), axis=0)
        indices = self.backend.arange(len(all_users))
        self.backend.random.shuffle(indices)
        return all_users[indices], all_items[indices]
        # neg_inds = []
        # for i in range(n_neg_samples):
        #     row = self.backend.random.randint(0, self.n_users, 1)[0]
        #     col = self.backend.random.randint(0, self.n_items, 1)[0]
        #     while self.interactions[row, col] > 0:
        #         row = self.backend.random.randint(0, self.n_users, 1)[0]
        #         col = self.backend.random.randint(0, self.n_items, 1)[0]
        #     neg_inds.append((row, col))
        # all_inds = list(zip(self.non_zero_rows[pos_inds], self.non_zero_columns[pos_inds])) + neg_inds
        # assert len(all_inds) == batch_size
        # all_inds = self.backend.array(all_inds)
        # self.backend.random.shuffle(all_inds)
        # return all_inds[:, 0], all_inds[:, 1]#, ratings
    
    # @njit(fastmath=True)
    def prime_sampler_state(self, n, exclude):
        """
        Initialize state to be used in fast sampler. Helps ensure excluded items are
        never sampled by placing them outside of sampling region.
        """
        # initialize typed numba dicts
        state = {n: n}
        state.pop(n)
        track = {n: n}
        track.pop(n)

        n_pos = n - len(state) - 1
        # reindex excluded items, placing them in the end
        for i, item in enumerate(exclude):
            pos = n_pos - i
            x = track.get(item, item)
            t = state.get(pos, pos)
            state[x] = t
            track[t] = x
            state.pop(pos, n)
            track.pop(item, n)
        return state
    
    # @njit(fastmath=True)
    def sample_fill(self, sample_size, sampler_state, remaining, result):
        """
        Sample a desired number of integers from a range (starting from zero)
        excluding black-listed elements defined in sample state. Used in
        conjunction with `prime_sample_state` method, which initializes state.
        Inspired by Fischer-Yates shuffle.
        """
        # gradually sample from the decreased size range
        for k in range(sample_size):
            i = randrange(remaining)
            result[k] = sampler_state.get(i, i)
            remaining -= 1
            sampler_state[i] = sampler_state.get(remaining, remaining)
            sampler_state.pop(remaining, -1)
    
    # @njit(parallel=True)
    def sample_element_wise(self, indptr, indices, n_cols, n_samples):
        """
        For every nnz entry of a CSR matrix, samples indices not present
        in its corresponding row.
        """
        result = np.empty((indptr[-1], n_samples), dtype=indices.dtype)
        for i in range(len(indptr) - 1):
            head = indptr[i]
            tail = indptr[i + 1]

            seen_inds = indices[head:tail]
            state = self.prime_sampler_state(n_cols, seen_inds)
            remaining = n_cols - len(seen_inds)
            # set_seed(seed_seq[i])
            for j in range(head, tail):
                sampler_state = state.copy()
                self.sample_fill(n_samples, sampler_state, remaining, result[j, :])
        return result