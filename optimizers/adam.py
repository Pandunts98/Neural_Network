import numpy as np

"""
In order to implement this, look at the **Algorithm 1** on page 2 of this paper:
(https://arxiv.org/pdf/1412.6980.pdf).
"""


def adam_optimizer(x, dx, config, state):
    state.setdefault('m_t', {})
    state.setdefault('v_t', {})
    t = 1
    for cur_layer_x, cur_layer_dx in zip(x, dx):
        for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):
            m_t = state['m_t'].setdefault(t, np.zeros_like(cur_dx))
            v_t = state['v_t'].setdefault(t, np.zeros_like(cur_dx))
            m_t = config['beta_1'] * m_t + (1 - config['beta_1']) * cur_dx
            v_t = config['beta_2'] * v_t + (1 - config['beta_2']) * (cur_dx ** 2)
            m_hat = m_t / (1 - (config['beta_1'] ** t))
            v_hat = v_t / (1 - (config['beta_2'] ** t))
            cur_old_grad = config['alpha'] * (m_hat / (np.sqrt(v_hat) - config["epsilon"]))
            state['m_t'][t] = m_t
            state['v_t'][t] = v_t

            np.add(cur_x, -cur_old_grad, out=cur_x)
            t += 1
