import _dynet as dy

class SelectiveBiGRU:
    def __init__(self, model, emb_dim, hid_dim):
        pc = model.add_subcollection()

        # BiGRU
        self.fGRUBuilder = dy.GRUBuilder(1, emb_dim, hid_dim, pc)
        self.bGRUBuilder = dy.GRUBuilder(1, emb_dim, hid_dim, pc)
        self.hid_dim = hid_dim

        # Selective Gate
        self._Ws = pc.add_parameters((2*hid_dim, 2*hid_dim))
        self._Us = pc.add_parameters((2*hid_dim, 2*hid_dim))
        self._bs = pc.add_parameters((2*hid_dim))

        self.pc   = pc
        self.spec = (emb_dim, hid_dim)

    def __call__(self, x_embs):
        x_len = len(x_embs)

        # BiGRU
        hf = dy.concatenate_cols(self.fGRUBuilder.initial_state().transduce(x_embs))
        hb = dy.concatenate_cols(self.bGRUBuilder.initial_state().transduce(x_embs[::-1])[::-1])
        h = dy.concatenate([hf, hb])

        # Selective Gate
        hb_1 = dy.pick(hb, index=0, dim=1)
        hf_n = dy.pick(hf, index=x_len-1, dim=1)
        s    = dy.concatenate([hb_1, hf_n])

        # Selection
        sGate = dy.logistic(dy.colwise_add(self.Ws*h, self.Us*s + self.bs))
        hp    = dy.cmult(h, sGate)

        return hp, hb_1

    def associate_parameters(self):
        self.Ws = dy.parameter(self._Ws)
        self.Us = dy.parameter(self._Us)
        self.bs = dy.parameter(self._bs)
        self.hf_0 = dy.zeroes((self.hid_dim))
        self.hb_0 = dy.zeroes((self.hid_dim))

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim = spec
        return SelectiveBiGRU(model, emb_dim, hid_dim)

    def param_collection(self):
        return self.pc

class AttentionalGRU:
    def __init__(self, model, emb_dim, hid_dim, maxout_dim, vocab_size):
        pc = model.add_subcollection()

        # GRU
        self.GRUBuilder = dy.GRUBuilder(1, emb_dim+2*hid_dim, hid_dim, pc)
        self._Wd = pc.add_parameters((hid_dim, hid_dim))
        self._bd = pc.add_parameters((hid_dim))
        self.hid_dim = hid_dim

        # Attention
        self._Wa = pc.add_parameters((hid_dim, hid_dim))
        self._Ua = pc.add_parameters((hid_dim, 2*hid_dim))
        self._va = pc.add_parameters((1, hid_dim))

        # Output
        self._Wr = pc.add_parameters((maxout_dim, hid_dim, emb_dim))
        self._Ur = pc.add_parameters((maxout_dim, hid_dim, 2*hid_dim))
        self._Vr = pc.add_parameters((maxout_dim, hid_dim, hid_dim))
        self._Wo = pc.add_parameters((vocab_size, hid_dim))

        self.pc   = pc
        self.spec = (emb_dim, hid_dim, maxout_dim, vocab_size)

    def __call__(self, x, tm1s=None, test=False):
        if test:
            # Initial states
            s_tm1 = tm1s[0]
            c_tm1 = tm1s[1]
            w_tm1 = x

            # GRU
            s_t = self.GRUBuilder.initial_state().set_s([s_tm1]).add_input(dy.concatenate([w_tm1, c_tm1])).output()

            # Attention
            e_t = dy.pick(self.va*dy.tanh(dy.colwise_add(self.Ua*self.hp, self.Wa*s_tm1)), 0)
            a_t = dy.softmax(e_t)
            #c_t = dy.esum(
            #    [dy.cmult(a_t_i, h_i) for a_t_i, h_i in zip(a_t, dy.transpose(self.hp))]
            #)
            c_t = self.hp*a_t # memory error?

            # Output
            r_t = dy.concatenate_cols(
                [Wr_j*w_tm1 + Ur_j*c_t + Vr_j*s_t for Wr_j, Ur_j, Vr_j in zip(self.Wr, self.Ur, self.Vr)]
            ) # Maxout
            m_t = dy.max_dim(r_t, d=1)
            y_t = dy.softmax(self.Wo*m_t)

            return s_t, c_t, y_t

        else:
            w_embs = x
            # Initial states
            s_tm1 = self.s_0
            c_tm1 = self.c_0
            GRU = self.GRUBuilder.initial_state().set_s([s_tm1])

            y = []
            for w_tm1 in w_embs:
                # GRU
                GRU = GRU.add_input(dy.concatenate([w_tm1, c_tm1]))
                s_t = GRU.output()

                # Attention
                e_t = dy.pick(self.va*dy.tanh(dy.colwise_add(self.Ua*self.hp, self.Wa*s_tm1)), 0)
                a_t = dy.softmax(e_t)
                #c_t = dy.esum(
                #    [dy.cmult(a_t_i, h_i) for a_t_i, h_i in zip(a_t, dy.transpose(self.hp))]
                #)
                c_t = self.hp*a_t # memory error?

                # Output
                r_t = dy.concatenate_cols(
                    [Wr_j*w_tm1 + Ur_j*c_t + Vr_j*s_t for Wr_j, Ur_j, Vr_j in zip(self.Wr, self.Ur, self.Vr)]
                ) # Maxout
                m_t = dy.max_dim(r_t, d=1)

                y_t = self.Wo*m_t
                y.append(y_t)

                # t -> tm1
                s_tm1 = s_t
                c_tm1 = c_t

            return y

    def associate_parameters(self):
        self.Wd = dy.parameter(self._Wd)
        self.bd = dy.parameter(self._bd)
        self.Wa = dy.parameter(self._Wa)
        self.Ua = dy.parameter(self._Ua)
        self.va = dy.parameter(self._va)
        self.Wr = dy.parameter(self._Wr)
        self.Ur = dy.parameter(self._Ur)
        self.Vr = dy.parameter(self._Vr)
        self.Wo = dy.parameter(self._Wo)

    def set_initial_states(self, hp, hb_1):
        self.s_0 = dy.tanh(self.Wd*hb_1 + self.bd)
        self.c_0 = dy.zeroes((2*self.hid_dim,))
        self.hp  = hp

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim, maxout_dim, vocab_size = spec
        return AttentionalGRU(model, emb_dim, hid_dim, maxout_dim, vocab_size)

    def param_collection(self):
        return self.pc
