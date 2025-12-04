import torch
import numpy as np
from functools import reduce
from copy import deepcopy
from torch.optim import Optimizer


def polyinterp(points, bounds=None):
    """
    ported from https://github.com/torch/optim/blob/master/polyinterp.lua

    Inputs:
        points (torch.tensor): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
    """
    x1, x2 = points[0, 0], points[1, 0]
    f1, f2 = points[0, 1], points[1, 1]
    g1, g2 = points[0, 2], points[1, 2]

    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


class LBFGS(Optimizer):
    """
    Implements the L-BFGS algorithm. Compatible with multi-batch and full-overlap
    L-BFGS implementations and (stochastic) Powell damping. Partly based on the 
    original L-BFGS implementation in PyTorch, Mark Schmidt's minFunc MATLAB code, 
    and Michael Overton's weak Wolfe line search MATLAB code.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 10/20/20.

    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'Wolfe': uses strong Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    References:
    [1] Berahas, Albert S., Jorge Nocedal, and Martin Takác. "A Multi-Batch L-BFGS 
        Method for Machine Learning." Advances in Neural Information Processing 
        Systems. 2016.
    [2] Bollapragada, Raghu, et al. "A Progressive Batching L-BFGS Method for Machine 
        Learning." International Conference on Machine Learning. 2018.
    [3] Lewis, Adrian S., and Michael L. Overton. "Nonsmooth Optimization via Quasi-Newton
        Methods." Mathematical Programming 141.1-2 (2013): 135-163.
    [4] Liu, Dong C., and Jorge Nocedal. "On the Limited Memory BFGS Method for 
        Large Scale Optimization." Mathematical Programming 45.1-3 (1989): 503-528.
    [5] Nocedal, Jorge. "Updating Quasi-Newton Matrices With Limited Storage." 
        Mathematics of Computation 35.151 (1980): 773-782.
    [6] Nocedal, Jorge, and Stephen J. Wright. "Numerical Optimization." Springer New York,
        2006.
    [7] Schmidt, Mark. "minFunc: Unconstrained Differentiable Multivariate Optimization 
        in Matlab." Software available at http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html 
        (2005).
    [8] Schraudolph, Nicol N., Jin Yu, and Simon Günter. "A Stochastic Quasi-Newton 
        Method for Online Convex Optimization." Artificial Intelligence and Statistics. 
        2007.
    [9] Wang, Xiao, et al. "Stochastic Quasi-Newton Methods for Nonconvex Stochastic 
        Optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.

    """

    def __init__(self, params, lr=1., history_size=10, line_search='Wolfe',
                 dtype=torch.float, debug=False):

        # ensure inputs are valid
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= history_size:
            raise ValueError("Invalid history size: {}".format(history_size))
        #if line_search not in ['Armijo', 'Wolfe', 'None']:
        if line_search != 'Wolfe':
            raise ValueError("Invalid line search: {}".format(line_search))

        defaults = dict(lr=lr, history_size=history_size, line_search=line_search, dtype=dtype, debug=debug)
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("L-BFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        state = self.state['global_state']
        state.setdefault('n_iter', 0)
        state.setdefault('curv_skips', 0)
        state.setdefault('H_diag',1)

        state['old_dirs'] = []
        state['old_stps'] = []

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_update(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            #p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            p.data.add_(update[offset : offset + numel].view_as(p.data), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _copy_params(self):
        current_params = []
        for param in self._params:
            current_params.append(deepcopy(param.data))
        return current_params

    def _load_params(self, current_params):
        i = 0
        for param in self._params:
            param.data[:] = current_params[i]
            i += 1

    def line_search(self, line_search):
        """
        Switches line search option.
        
        Inputs:
            line_search (str): designates line search to use
                Options:
                    'Wolfe': uses Armijo-Wolfe bracketing line search
        
        """
        
        group = self.param_groups[0]
        group['line_search'] = line_search
        
        return

    def two_loop_recursion(self, vec):
        """
        Performs two-loop recursion on given vector to obtain Hv.

        Inputs:
            vec (tensor): 1-D tensor to apply two-loop recursion to

        Output:
            r (tensor): matrix-vector product Hv

        """

        group = self.param_groups[0]
        history_size = group['history_size']

        state = self.state['global_state']
        old_dirs = state.get('old_dirs')    # change in gradients
        old_stps = state.get('old_stps')    # change in iterates
        H_diag = state.get('H_diag')

        # compute the product of the inverse Hessian approximation and the gradient
        num_old = len(old_dirs)

        if 'rho' not in state:
            state['rho'] = [None] * history_size
            state['alpha'] = [None] * history_size
        rho = state['rho']
        alpha = state['alpha']

        for i in range(num_old):
            rho[i] = 1. / old_stps[i].dot(old_dirs[i])

        q = vec
        for i in range(num_old - 1, -1, -1):
            alpha[i] = old_dirs[i].dot(q) * rho[i]
            q.add_(old_stps[i], alpha= -alpha[i])

        # multiply by initial Hessian 
        # r/d is the final direction
        r = torch.mul(q, H_diag)
        for i in range(num_old):
            beta = old_stps[i].dot(r) * rho[i]
            r.add_(old_dirs[i], alpha= alpha[i]-beta)

        return r

    def curvature_update(self, flat_grad, eps=1e-10, damping=False):
        """
        Performs curvature update.

        Inputs:
            flat_grad (tensor): 1-D tensor of flattened gradient for computing 
                gradient difference with previously stored gradient
            eps (float): constant for curvature pair rejection or damping (default: 1e-10)
            damping (bool): flag for using Powell damping (default: False)
        """

        assert len(self.param_groups) == 1

        # load parameters
        if(eps <= 0):
            raise(ValueError('Invalid eps; must be positive.'))

        group = self.param_groups[0]
        history_size = group['history_size']
        debug = group['debug']

        # variables cached in state (for tracing)
        state = self.state['global_state']
                 
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')

        # compute y's
        y = flat_grad.sub(prev_flat_grad)
        s = d.mul(t)
        ys = y.dot(s)  # y*s

        # update L-BFGS matrix
        if ys > eps  :

            # updating memory
            if len(old_dirs) == history_size:
                # shift history by one (limited-memory)
                old_dirs.pop(0)
                old_stps.pop(0)

            # store new direction/step
            old_dirs.append(s)
            old_stps.append(y)

            # update scale of initial Hessian approximation
            H_diag = ys / y.dot(y)  # (y*y)
            
            state['old_dirs'] = old_dirs
            state['old_stps'] = old_stps
            state['H_diag'] = H_diag

        else:
            # save skip
            state['curv_skips'] += 1
            if debug:
                print('Curvature pair skipped due to failed criterion')


        return

    def _step(self, p_k, g_Ok, options=None):
        """
        Performs a single optimization step.

        Inputs:
            p_k (tensor): 1-D tensor specifying search direction
            g_Ok (tensor): 1-D tensor of flattened gradient over overlap O_k used
                            for gradient differencing in curvature pair update
            options (dict): contains options for performing line search (default: None)

        Options for Wolfe line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
            'c2' (float): curvature condition constant in (0, 1) (default: 0.9)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'ls_debug' (bool): debugging mode for line search

        Outputs (Only strong Wolfe implemented):
          . Wolfe line search:
                F_new (tensor): loss function at new iterate
                g_new (tensor): gradient at new iterate
                t (float): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                grad_eval (int): number of gradient evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function

        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

        """

        if options is None:
            options = {}
        assert len(self.param_groups) == 1

        # load parameter options
        group = self.param_groups[0]
        lr = group['lr']
        line_search = group['line_search']
        debug = group['debug']

        # variables cached in state (for tracing)
        state = self.state['global_state']
        d = state.get('d')
        t = state.get('t')
        prev_flat_grad = state.get('prev_flat_grad')

        # keep track of nb of iterations
        state['n_iter'] += 1

        # set search direction
        d = p_k

        # modify previous gradient
        if prev_flat_grad is None:
            prev_flat_grad = g_Ok.clone(memory_format=torch.contiguous_format)
        else:
            prev_flat_grad.copy_(g_Ok)

        # set initial step size
        #t = lr
        # reset initial guess for step size
        if state["n_iter"] == 1:
            t = float(min(1.0, 1.0 / g_Ok.abs().sum()) * lr)
        else:
            t = lr

        # closure evaluation counter
        closure_eval = 0

        # ----------------------------------
        # === Strong Wolfe line search ===
        # ----------------------------------

        # load options
        if options:
            if 'closure' not in options.keys():
                raise(ValueError('closure option not specified.'))
            else:
                closure = options['closure']

            if 'current_loss' not in options.keys():
                F_k = closure()
                closure_eval += 1
            else:
                F_k = options['current_loss']

            if 'gtd' not in options.keys():
                gtd = g_Ok.dot(d)
            else:
                gtd = options['gtd']

            if 'c1' not in options.keys():
                c1 = 1e-4
            elif options['c1'] >= 1 or options['c1'] <= 0:
                raise(ValueError('Invalid c1; must be strictly between 0 and 1.'))
            else:
                c1 = options['c1']

            if 'c2' not in options.keys():
                c2 = 0.9
            elif options['c2'] >= 1 or options['c2'] <= 0:
                raise(ValueError('Invalid c2; must be strictly between 0 and 1.'))
            elif options['c2'] <= c1:
                raise(ValueError('Invalid c2; must be strictly larger than c1.'))
            else:
                c2 = options['c2']

            if 'max_ls' not in options.keys():
                max_ls = 10
            elif options['max_ls'] <= 0:
                raise(ValueError('Invalid max_ls; must be positive.'))
            else:
                max_ls = options['max_ls']
  
            if 'ls_debug' not in options.keys():
                ls_debug = False
            else:
                ls_debug = options['ls_debug']

        else:
            raise(ValueError('Options are not specified; need closure evaluating function.'))

        # initialize counters
        ls_step = 0
        grad_eval = 0 # tracks gradient evaluations
        t_prev = 0 # old steplength

        # ---------------------------
        # === BRACKETING PHASE ===
        # ---------------------------

        # begin print for debug mode
        if ls_debug:
            print('==================================== Begin Wolfe line search ====================================')
            print('F(x): %.8e  g*d: %.8e' % (F_k, gtd))

        # check if search direction is descent direction
        if gtd >= 0:
            desc_dir = False
            if debug:
                print('Not a descent direction!')
        else:
            desc_dir = True

        # store values 
        current_params = self._copy_params()

        # update and evaluate at new point
        self._add_update(t, d)
        F_new = closure(); closure_eval += 1
        # compute gradient 
        F_new.backward()
        g_new = self._gather_flat_grad(); grad_eval += 1
        gtd_new = g_new.dot(d)

        # We'll try to bracket a step satisfying strong Wolfe, or an interval containing it.
        bracket = None
        bracket_f = None
        bracket_g = None
        bracket_gtd = None
        done = False

        # keep previous values for bracketing logic
        t_prev = 0.0
        F_prev = F_k
        gtd_prev = gtd  # directional derivative at alpha=0
        g_prev_vec = prev_flat_grad # initial gradient vector

        # main loop
        while True:

            # print info if debugging
            if ls_debug:
                print('LS Step: %d  t: %.8e  t_prev: %.8e' 
                        % (ls_step, t, t_prev))

            # check if maximum number of line search steps have been reached
            if ls_step == max_ls:

                bracket = [0, t]
                bracket_f = [F_k, F_new]
                bracket_g = [prev_flat_grad, g_new]
                break

            # print info if debugging
            if ls_debug:
                print('Armijo:  F(x+td): %.8e  F+c1*t*g*d: %.8e  F(x): %.8e'
                        % (F_new, F_k + c1 * t * gtd, F_k))

            # Condition 1: Armijo violation OR (curvature not bad but function increased wrt previous)
            # Note: we compare F_new with F_prev to detect non-monotone behaviour that indicates bracket
            if (F_new > F_k + c1 * t * gtd) or (ls_step > 1 and F_new >= F_prev):
                # bracket found between t_prev and t
                bracket = [t_prev, t]
                bracket_f = [F_prev, F_new]
                bracket_g = [g_prev_vec, g_new.clone(memory_format=torch.contiguous_format)]
                bracket_gtd = [gtd_prev, gtd_new]
                break
            

            # print info if debugging
            if ls_debug:
                print('Wolfe 1: |g(x+td)*d|: %.8e  -c2*g*d: %.8e  gtd: %.8e'
                        % (abs(gtd_new), -c2 * gtd, gtd))
        
            # Condition 2: strong Wolfe satisfied at t 
            if abs(gtd_new) <= -c2 * gtd:
                # found acceptable t (single-point "bracket")
                bracket = [t]
                bracket_f = [F_new]
                bracket_g = [g_new]
                done = True
                break


            # print info if debugging
            if ls_debug:
                print('Wolfe 2: g(x+td)*d: %.8e'
                        % (gtd_new))
                
            # Condition 3: derivative >= 0 -> bracket found between t_prev and t
            if gtd_new >= 0:
                bracket = [t_prev, t]
                bracket_f = [F_prev, F_new]
                bracket_g = [g_prev_vec, g_new.clone(memory_format=torch.contiguous_format)]
                bracket_gtd = [gtd_prev, gtd_new]
                break

                
            # otherwise, advance: extrapolate or interpolate to get new t
            # compute safe interpolation bounds
            min_step = t + 0.01 * (t - t_prev) 
            max_step = t * 10.0 
            tmp = t

            # compute new steplength: interpolate between a and b
            # print info if debugging
            if ls_debug:
                print('Interpolation: [t_prv; t]: [%.4e, %.4e]  [F_prev; F_new]: [%.4e, %.4e]  [gtd_prev; gtd_new]: [%.4e, %.4e]'
                        % (t_prev, t, F_prev.item(), F_new.item(), gtd_prev.item(), gtd_new.item()))

            t_candidate  = polyinterp(torch.tensor([[t_prev, F_prev.item(), gtd_prev.item()], 
                                                    [t, F_new.item(), gtd_new.item()]]),
                                                    bounds=(min_step,max_step))
            t_candidate = max(min_step, min(max_step, t_candidate))

            # if we obtain nonsensical value from interpolation
            if t_candidate <= 0 or np.isnan(t_candidate):
                print("Problematic t_candidate detected! ", t_candidate)

            t = float(t_candidate)

            # store previous values for next iteration
            t_prev = tmp
            F_prev = F_new
            gtd_prev = gtd_new
            g_prev_vec = g_new.clone(memory_format=torch.contiguous_format)

            # update parameters
            self._load_params(current_params)
            self._add_update(t, d)

            # evaluate closure
            F_new = closure(); closure_eval += 1
            # compute gradient 
            F_new.backward()
            g_new = self._gather_flat_grad(); grad_eval += 1
            gtd_new = g_new.dot(d)

            ls_step += 1

        # ---------------------------
        # === ZOOM / REFINEMENT PHASE ===
        # ---------------------------
        # If bracket contains two endpoints, refine between them. If bracket is single-point (done earlier),
        # we accept that point. This mirrors PyTorch behaviour (zoom loop).
        if ls_debug:
            print('==================================== Beging zoom phase ===================================')

        if len(bracket) == 1:
            # single point satisfied Wolfe already
            low_pos = 0

            if ls_debug:
                print("Single element bracket: Second condition satisfied!")

        else:
            # bracket is [lo, hi]
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)

            if ls_debug:
                    print("Two elements bracket: Braket around the satisfying point:")
                    print('bracket: [%.4e, %.4e]  bracket_f: [%.4e, %.4e]'
                            % (bracket[low_pos], bracket[high_pos], 
                               bracket_f[low_pos].item(), bracket_f[high_pos].item()))

            insuf_progress = False

            # zoom loop (bounded by max_ls)
            while not done and ls_step < max_ls:
                # small bracket check (safety)
                d_norm = d.abs().max()
                if abs(bracket[1] - bracket[0]) * d_norm < 1e-10:
                    break

                # cubic interpolation inside [lo, hi]
                t_candidate = polyinterp(torch.tensor([[bracket[0], bracket_f[0].item(),  bracket_gtd[0].item()], 
                                                       [bracket[1], bracket_f[1].item(),  bracket_gtd[1].item()]]))
                
                # if we obtain nonsensical value from interpolation
                if t_candidate <= 0 or (np.isnan(t_candidate)):
                    print('Problematic t_candidate detected!', t_candidate)         

                # ensure candidate not too close to boundaries
                eps = 0.1 * (max(bracket) - min(bracket))
                if min(max(bracket) - t_candidate, t_candidate - min(bracket)) < eps:
                    if insuf_progress or t >= max(bracket) or t <= min(bracket):
                        # move 0.1 away from nearest boundary
                        if abs(t_candidate - max(bracket)) < abs(t_candidate - min(bracket)):
                            t_candidate = max(bracket) - eps
                        else:
                            t_candidate = min(bracket) + eps
                        insuf_progress = False
                    else:
                        insuf_progress = True
                else:
                    insuf_progress = False

                t = float(t_candidate)

                if ls_debug:
                    print("Zooming step: ", ls_step)
                    print('bracket: [%.4e, %.4e]  bracket_f: [%.4e, %.4e]  bracket_gtd: [%.4e, %.4e]'
                            % (bracket[low_pos], bracket[high_pos], 
                            bracket_f[low_pos].item(), bracket_f[high_pos].item(), 
                            bracket_gtd[low_pos].item(), bracket_gtd[high_pos].item()))
                    print("t_candidate", t_candidate)

                # apply step
                self._load_params(current_params)
                self._add_update(t, d)

                # evaluate new point
                F_new = closure(); closure_eval += 1
                F_new.backward()
                g_new = self._gather_flat_grad(); grad_eval += 1
                gtd_new = g_new.dot(d)
                ls_step += 1

                # print info if debugging
                if ls_debug:
                    print('Armijo:  F(x+td): %.8e  F+c1*t*g*d: %.8e  F(x): %.8e'
                            % (F_new, F_k + c1 * t * gtd, F_k))

                # check Armijo or whether new point is not better than low
                if (F_new > F_k + c1 * t * gtd) or (F_new >= bracket_f[low_pos]):                       
                    # Armijo condition not satisfied or not lower than lowest point
                    bracket[high_pos] = t #hi = t
                    bracket_f[high_pos] = F_new #Fhi = float(F_new)
                    bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)   #grad_hi_vec = g_new.clone()
                    bracket_gtd[high_pos] = gtd_new #gtd_hi = float(gtd_new)
                    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)

                else:
                    # print info if debugging
                    if ls_debug:
                        print('Wolfe 1: |g(x+td)*d|: %.8e  -c2*g*d: %.8e  gtd: %.8e'
                                % (abs(gtd_new), -c2 * gtd, gtd))
                        
                    # check curvature
                    if abs(gtd_new) <= -c2 * gtd:
                        done = True

                    # if gtd_new*(hi-lo) >= 0, move hi to lo (flip)
                    elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:

                        # old high becomes new low
                        bracket[high_pos] = bracket[low_pos] #hi = lo
                        bracket_f[high_pos] = bracket_f[low_pos] #Fhi = Flo
                        bracket_g[high_pos] = bracket_g[low_pos]  #grad_hi_vec = grad_lo_vec.clone() if grad_lo_vec is not None else None
                        bracket_gtd[high_pos] = bracket_gtd[low_pos] #gtd_hi = gtd_lo

                    # new point becomes new low
                    bracket[low_pos] = t  #lo = t
                    bracket_f[low_pos] = F_new #Flo = float(F_new)
                    bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  #grad_lo_vec = g_new.clone()
                    bracket_gtd[low_pos] = gtd_new  #gtd_lo = float(gtd_new)

        if ls_debug:
            print('===================================== End zoom phase =====================================')

        # return stuff
        t = bracket[low_pos]  
        F_new = bracket_f[low_pos]
        g_new = bracket_g[low_pos]  

        # debug prints (unchanged)
        if ls_debug:
            print('Final Steplength:', t)
            print('===================================== End Wolfe line search =====================================')

        # save state & return (unchanged)
        state['d'] = d
        state['prev_flat_grad'] = prev_flat_grad
        state['t'] = t

        return F_new, g_new, t, ls_step, closure_eval, grad_eval, desc_dir             


class FullBatchLBFGS(LBFGS):
    """
    Implements full-batch or deterministic L-BFGS algorithm. Compatible with
    Powell damping. Can be used when evaluating a deterministic function and
    gradient. Wraps the LBFGS optimizer. Performs the two-loop recursion,
    updating, and curvature updating in a single step.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 11/15/18.

    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    """

    def __init__(self, params, lr=1, history_size=10, line_search='Wolfe', 
                 dtype=torch.float, debug=False):
        super(FullBatchLBFGS, self).__init__(params, lr, history_size, line_search, 
             dtype, debug)

    def step(self, options=None):
        """
        Performs a single optimization step.

        Inputs:
            options (dict): contains options for performing line search (default: None)
            
        General Options:
            'eps' (float): constant for curvature pair rejection (default: 1e-10)

        Options for Wolfe line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
            'c2' (float): curvature condition constant in (0, 1) (default: 0.9)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'ls_debug' (bool): debugging mode for line search

        Outputs (only Wolfe line search implemented):
          . Wolfe line search:
                F_new (tensor): loss function at new iterate
                g_new (tensor): gradient at new iterate
                t (float): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                grad_eval (int): number of gradient evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded

        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

        """
        
        # load options for damping and eps
        if 'damping' not in options.keys():
            damping = False
        else:
            damping = options['damping']
            
        if 'eps' not in options.keys():
            eps = 1e-10
        else:
            eps = options['eps']
        
        # gather gradient
        grad = self._gather_flat_grad()
        
        # update curvature if after 1st iteration
        state = self.state['global_state']
        if state['n_iter'] > 0:
            self.curvature_update(grad, eps, damping)

        # compute search direction
        p = self.two_loop_recursion(-grad)

        # take step
        self._step(p, grad, options=options)

        return