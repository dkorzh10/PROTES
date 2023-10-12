import jax
import jax.numpy as jnp
import optax
from time import perf_counter as tpc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.functional import softmax, log_softmax
import numpy as np


def protes_gpt(f, model, tokenizer, d, n=10, m=None, n0=34, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, seed=0,
           is_max=False, log=False, info={}, P=None,
           with_info_i_opt_list=False, with_info_full=False):
#     assert n == 10
    assert k == 1
    

    time = tpc()
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'k_top': k_top,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': [], "ll_list": []})
    if with_info_full:
        info.update({
            'P_list': [], 'I_list': [], 'y_list': []})

    rng = jax.random.PRNGKey(seed)

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(k, d, n, key)

    optim = optax.adam(lr)
    state = optim.init(P)

#     sample = jax.jit(jax.vmap(_sample, (None, 0)))
    sample = jax.jit(jax.vmap(_sample, (0, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood, (None, 0)))


    @jax.jit
    def loss(P_cur, I_cur):
        l = likelihood(P_cur, I_cur)
        return jnp.mean(-l)

    loss_grad = jax.grad(loss)

    @jax.jit
    def optimize(state, P_cur, I_cur):
        grads = loss_grad(P_cur, I_cur)
        updates, state = optim.update(grads, state)
        P_cur = jax.tree_util.tree_map(lambda p, u: p + u, P_cur, updates)
        return state, P_cur

#     I = sample(P[:, 0], jax.random.split(key, k))
    I = sample(P[0], jax.random.split(key, d))
    I = I[None, :]
    while True:
        q = np.array(I + n0)  # [None, :]
        q = torch.tensor(q).to(model.device)
        with torch.no_grad():
            logits = model(q).logits
        P = softmax(logits[:, :, n0:n0+n], dim=-1)
        # batch x d x n

        rng, key = jax.random.split(rng)
        P = jnp.array(P.cpu().detach().numpy())

#         I = sample(P[:, 0], jax.random.split(key, k))
        
        I = sample(P[0], jax.random.split(key, d))[None, :]
        y = f(I)
        if y is None:
            break

        y = jnp.array(y)
        info['m'] += y.shape[0]
        l = likelihood(P, I)
        l = jnp.mean(-l)
        info["ll_list"].append(l)
        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)

        if info['m_max'] and info['m'] >= info['m_max']:
            break

        ind = jnp.argsort(y, kind='stable')
        ind = (ind[::-1] if is_max else ind)[:k_top]
        for _ in range(k_gd):
            state, P = optimize(state, P, I[ind, :])
            
        
        info['t'] = tpc() - time
        _log(info, log, is_new)

    info['t'] = tpc() - time
    _log(info, log, is_new, is_end=True)

    return info['i_opt'], info['y_opt'], info["ll_list"]


def _likelihood(P, I):

    """Compute the likelihood of sequence i for decicoder model."""
    #     y = jnp.cumsum(jax.nn.log_softmax(P, axis=-1), axis=1) #[:, -1]
    y = jax.nn.log_softmax(P, axis=-1)
    probs = jnp.zeros_like(I)

    # Y from 0 to 9
    # try to swap this cycle with jax.lax.scan
    for i in range(len(I)):
        probs += y[:, i, I[i]]

    probs = jnp.array(probs)
    return probs



def _generate_initial(k, d, n,  key):
    keyl, _ = jax.random.split(key, 2)
    P = jax.random.uniform(keyl, (k, d, n))
    return P

# def _sample(P, key):
#     print(P[0])
#     i = jax.random.choice(key, jnp.arange(P.shape[-1]), p=P[0], shape=(1,))
#     return i
# sample = jax.jit(jax.vmap(_sample, (0, None)))


def _sample(P, key):
    i = jax.random.choice(key, P.shape[-1], p=P)
    return i
# sample = jax.jit(jax.vmap(_sample, (0, 0)))

# jax.debug



def _log(info, log=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if is_end:
        text += ' <<< DONE'

    print(text)


def _process(P, I, y, info, with_info_i_opt_list, with_info_full):
    """Check the current batch of function values and save the improvement."""
    ind_opt = jnp.argmax(y) if info['is_max'] else jnp.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr

    if is_new or with_info_full:
        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(info['y_opt'])

        if with_info_i_opt_list or with_info_full:
            info['i_opt_list'].append(info['i_opt'].copy())

    if with_info_full:
        info['P_list'].append([G.copy() for G in P])
        info['I_list'].append(I.copy())
        info['y_list'].append(y.copy())

    return is_new



