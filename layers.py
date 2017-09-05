import _dynet as dy

class SelectiveBiGRU:
    def __init__(self, emb_dim, hid_dim, model):
        # BiGRU
        self.fGRU_Builder = dy.GRUBuilder(1, emb_dim, hid_dim, model)
        self.bGRU_Builder = dy.GRUBuilder(1, emb_dim, hid_dim, model)

        # Selective gate
        self._W_s = model.add_parameters((hid_dim*2, hid_dim*2))
        self._U_s = model.add_parameters((hid_dim*2, hid_dim*2))
        self._b_s = model.add_parameters((hid_dim*2))

    def associate_parameters(self):
        self.W_s = dy.parameter(self._W_s)
        self.U_s = dy.parameter(self._U_s)
        self.b_s = dy.parameter(self._b_s)

    def f_prop(self, x):
        self.h_f = self.fGRU_Builder.initial_state().transduce(x)
        self.h_b = self.bGRU_Builder.initial_state().transduce(x[::-1])[::-1]
        h = [dy.concatenate([h_f_i, h_b_i]) for h_f_i, h_b_i in zip(self.h_f, self.h_b)]

        s = dy.concatenate([self.h_b[0], self.h_f[-1]])
        s_gate = [dy.logistic(self.W_s*h_i + self.U_s*s + self.b_s) for h_i in h]
        h_gated = [dy.cmult(h_i, s_gate_i) for h_i, s_gate_i in zip(h, s_gate)]
        return h_gated

class AttentionalGRU:
    def __init__(self, emb_dim, hid_dim, maxout_dim, vocab_size, model):
        # GRU
        self._W_d = model.add_parameters((hid_dim, hid_dim))
        self._b_d = model.add_parameters((hid_dim))
        self.GRU_Builder = dy.GRUBuilder(1, emb_dim+hid_dim*2, hid_dim, model)
        # Attention layer
        self._W_a = model.add_parameters((hid_dim, hid_dim))
        self._U_a = model.add_parameters((hid_dim, hid_dim*2))
        self._v_a = model.add_parameters((hid_dim))

        # Output layer
        self._W_r = model.add_parameters((maxout_dim, hid_dim, emb_dim))
        self._U_r = model.add_parameters((maxout_dim, hid_dim, hid_dim*2))
        self._V_r = model.add_parameters((maxout_dim, hid_dim, hid_dim))
        self._W_o = model.add_parameters((vocab_size, hid_dim))

        # Dims
        self.hid_dim = hid_dim

    def associate_parameters(self):
        self.W_d = dy.parameter(self._W_d)
        self.b_d = dy.parameter(self._b_d)
        self.W_a = dy.parameter(self._W_a)
        self.U_a = dy.parameter(self._U_a)
        self.v_a = dy.parameter(self._v_a)
        self.W_r = dy.parameter(self._W_r)
        self.U_r = dy.parameter(self._U_r)
        self.V_r = dy.parameter(self._V_r)
        self.W_o = dy.parameter(self._W_o)

    def set_init_states(self, h, h_b_0):
        self.c_0 = dy.zeroes((self.hid_dim*2,))
        self.s_0 = dy.tanh(self.W_d*h_b_0 + self.b_d)
        self.h = h

    def f_prop(self, x):
        # Initial states
        c_tm1 = self.c_0
        s_tm1 = self.s_0
        GRU = self.GRU_Builder.initial_state([s_tm1])

        y = []
        for x_tm1 in x:
            # GRU
            s_t_state = GRU.add_input(dy.concatenate([x_tm1, c_tm1]))
            s_t = s_t_state.output()

            # Attention
            e_t = dy.concatenate(
                [dy.dot_product(self.v_a, dy.tanh(self.W_a*s_tm1 + self.U_a*h_t)) for h_t in self.h]
            )
            a_t = dy.softmax(e_t)
            c_t = dy.esum(
                [dy.cmult(a_t_i, h_i) for a_t_i, h_i in zip(a_t, self.h)]
            )

            # Output
            r_t = dy.concatenate([W_r_j*x_tm1 for W_r_j in self.W_r], d=1) \
                + dy.concatenate([U_r_j*c_t   for U_r_j in self.U_r], d=1) \
                + dy.concatenate([V_r_j*s_t   for V_r_j in self.V_r], d=1)
            m_t = dy.max_dim(r_t, d=1) # Maxout

            y_t = self.W_o*m_t
            y.append(y_t)

            # t -> tm1
            s_tm1 = s_t
            c_tm1 = c_t

        return y
