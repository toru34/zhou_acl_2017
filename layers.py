import _dynet as dy

class SelectiveBiGRU:
    def __init__(self, emb_dim, hid_dim, model):
        pc = model.add_subcollection()

        # BiGRU
        self.fGRUBuilder = dy.GRUBuilder(1, emb_dim, hid_dim, pc)
        self.bGRUBuilder = dy.GRUBuilder(1, emb_dim, hid_dim, pc)

        # Selective gate
        self._W_s = pc.add_parameters((2*hid_dim, 2*hid_dim))
        self._U_s = pc.add_parameters((2*hid_dim, 2*hid_dim))
        self._b_s = pc.add_parameters((2*hid_dim))

        self.pc = pc
        self.spec = (emb_dim, hid_dim)

    def __call__(self, x):
        self.h_f = self.fGRUBuilder.initial_state().transduce(x)
        self.h_b = self.bGRUBuilder.initial_state().transduce(x[::-1])[::-1]
        h = [dy.concatenate([h_f_i, h_b_i]) for h_f_i, h_b_i in zip(self.h_f, self.h_b)]

        s = dy.concatenate([self.h_b[0], self.h_f[-1]])
        s_gate = [dy.logistic(self.W_s*h_i + self.U_s*s + self.b_s) for h_i in h]
        h_gated = [dy.cmult(h_i, s_gate_i) for h_i, s_gate_i in zip(h, s_gate)]
        return h_gated

    def associate_parameters(self):
        self.W_s = dy.parameter(self._W_s)
        self.U_s = dy.parameter(self._U_s)
        self.b_s = dy.parameter(self._b_s)

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim = spec
        return SelectiveBiGRU(emb_dim, hid_dim, model)

    def param_collection(self):
        return self.pc

class AttentionalGRU:
    def __init__(self, emb_dim, hid_dim, maxout_dim, vocab_size, model):
        pc = model.add_subcollection()

        # GRU
        self.GRUBuilder = dy.GRUBuilder(1, emb_dim+2*hid_dim, hid_dim, pc)
        self._W_d = pc.add_parameters((hid_dim, hid_dim))
        self._b_d = pc.add_parameters((hid_dim))

        # Attention
        self._W_a = pc.add_parameters((hid_dim, hid_dim))
        self._U_a = pc.add_parameters((hid_dim, 2*hid_dim))
        self._v_a = pc.add_parameters((hid_dim))

        # Output
        self._W_r = pc.add_parameters((maxout_dim, hid_dim, emb_dim))
        self._U_r = pc.add_parameters((maxout_dim, hid_dim, 2*hid_dim))
        self._V_r = pc.add_parameters((maxout_dim, hid_dim, hid_dim))
        self._W_o = pc.add_parameters((vocab_size, hid_dim))

        self.hid_dim = hid_dim
        self.pc = pc
        self.spec = (emb_dim, hid_dim, maxout_dim, vocab_size)

    def __call__(self, x, tm1s=None, test=False):
        if test:
            x_tm1 = x
            s_tm1 = tm1s[0]
            c_tm1 = tm1s[1]
            GRU = self.GRUBuilder.initial_state([s_tm1])

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

            y_t = dy.softmax(self.W_o*m_t)

            return s_t, c_t, y_t

        else:
            # Initial states
            s_tm1 = self.s_0
            c_tm1 = self.c_0
            GRU = self.GRUBuilder.initial_state([s_tm1])

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

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim, maxout_dim, vocab_size = spec
        return AttentionalGRU(emb_dim, hid_dim, maxout_dim, vocab_size, model)

    def param_collection(self):
        return self.pc

    def set_init_states(self, h, h_b_0):
        self.s_0 = dy.tanh(self.W_d*h_b_0 + self.b_d)
        self.c_0 = dy.zeroes((2*self.hid_dim,))
        self.h = h
