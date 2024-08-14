import qutip.core.data as _data
import jax.numpy as np

class MCIntegrator:
    """
    Integrator like object for mcsolve trajectory.
    """
    name = "mcsolve"

    def __init__(self, integrator, system, options=None):
        self._integrator = integrator
        self.system = system
        self._c_ops = system.c_ops
        self._n_ops = system.n_ops
        self.options = options
        self._generator = None
        self.method = f"{self.name} {self._integrator.method}"
        self._is_set = False
        self.issuper = self._c_ops[0].issuper

    def set_state(self, t, state0, generator,
                  no_jump=False, jump_prob_floor=0.0):
        """
        Set the state of the ODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        generator : numpy.random.generator
            Random number generator.

        no_jump: Bool
            whether or not to sample the no-jump trajectory.
            If so, the "random number" should be set to zero

        jump_prob_floor: float
            if no_jump == False, this is set to the no-jump
            probability. This setting ensures that we sample
            a trajectory with jumps
        """
        self.collapses = []
        self.system._register_feedback("CollapseFeedback", self.collapses)
        self._generator = generator
        if no_jump:
            self.target_norm = 0.0
        else:
            self.target_norm = (
                self._generator.random() * (1 - jump_prob_floor)
                + jump_prob_floor
            )
        self._integrator.set_state(t, state0)
        self._is_set = True

    def get_state(self, copy=True):
        return *self._integrator.get_state(copy), self._generator

    def integrate(self, t, copy=False):
        t_old, y_old = self._integrator.get_state(copy=False)
        norm_old = self._prob_func(y_old)
        while t_old < t:
            t_step, state = self._integrator.mcstep(t, copy=False)
            norm = self._prob_func(state)
            if norm <= self.target_norm:
                t_col, state = self._find_collapse_time(norm_old, norm,
                                                        t_old, t_step)
                self._do_collapse(t_col, state)
                t_old, y_old = self._integrator.get_state(copy=False)
                norm_old = 1.
            else:
                t_old, y_old = t_step, state
                norm_old = norm

        return t_old, _data.mul(y_old, 1 / self._norm_func(y_old))

    def run(self, tlist):
        for t in tlist[1:]:
            yield self.integrate(t, False)

    def reset(self, hard=False):
        self._integrator.reset(hard)

    def _prob_func(self, state):
        if self.issuper:
            return _data.trace_oper_ket(state).real
        return _data.norm.l2(state)**2

    def _norm_func(self, state):
        if self.issuper:
            return _data.trace_oper_ket(state).real
        return _data.norm.l2(state)

    def _find_collapse_time(self, norm_old, norm, t_prev, t_final):
        """Find the time of the collapse and state just before it."""
        tries = 0
        while tries < self.options['norm_steps']:
            tries += 1
            if (t_final - t_prev) < self.options['norm_t_tol']:
                t_guess = t_final
                _, state = self._integrator.get_state()
                break
            t_guess = (
                t_prev
                + ((t_final - t_prev)
                   * np.log(norm_old / self.target_norm)
                   / np.log(norm_old / norm))
            )
            if (t_guess - t_prev) < self.options['norm_t_tol']:
                t_guess = t_prev + self.options['norm_t_tol']
            _, state = self._integrator.mcstep(t_guess, copy=False)
            norm2_guess = self._prob_func(state)
            if (
                np.abs(self.target_norm - norm2_guess) <
                self.options['norm_tol'] * self.target_norm
            ):
                break
            elif (norm2_guess < self.target_norm):
                # t_guess is still > t_jump
                t_final = t_guess
                norm = norm2_guess
            else:
                # t_guess < t_jump
                t_prev = t_guess
                norm_old = norm2_guess

        if tries >= self.options['norm_steps']:
            raise RuntimeError(
                "Could not find the collapse time within desired tolerance. "
                "Increase accuracy of the ODE solver or lower the tolerance "
                "with the options 'norm_steps', 'norm_tol', 'norm_t_tol'.")

        return t_guess, state

    def _do_collapse(self, collapse_time, state):
        """
        Do the collapse:
        - Find which operator did the collapse.
        - Update the state and Integrator.
        - Next collapse norm location
        - Store collapse info.
        """
        # collapse_time, state is at the collapse
        if len(self._n_ops) == 1:
            which = 0
        else:
            probs = np.zeros(len(self._n_ops))
            for i, n_op in enumerate(self._n_ops):
                probs[i] = n_op.expect_data(collapse_time, state).real
            probs = np.cumsum(probs)
            which = np.searchsorted(probs,
                                    probs[-1] * self._generator.random())

        state_new = self._c_ops[which].matmul_data(collapse_time, state)
        new_norm = self._norm_func(state_new)
        if new_norm < self.options['mc_corr_eps']:
            # This happen when the collapse is caused by numerical error
            state_new = _data.mul(state, 1 / self._norm_func(state))
        else:
            state_new = _data.mul(state_new, 1 / new_norm)
            self.collapses.append((collapse_time, which))
            # this does not need to be modified for improved sampling:
            # as noted in Abdelhafez PRA (2019),
            # after a jump we reset to the full range [0, 1)
            self.target_norm = self._generator.random()
        self._integrator.set_state(collapse_time, state_new)

    def arguments(self, args):
        if args:
            self._integrator.arguments(args)
            for c_op in self._c_ops:
                c_op.arguments(args)
            for n_op in self._n_ops:
                n_op.arguments(args)

    @property
    def integrator_options(self):
        return self._integrator.integrator_options
