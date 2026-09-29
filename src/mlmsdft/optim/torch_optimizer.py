
# -*- coding: utf-8 -*-
""" torch wrapper around minimize(...) function"""
import numpy as np

import torch

from mlmsdft.optim.minimize import minimize


class OptimizerTelemetry:
    """Lightweight counters for optimizer host synchronization points."""

    def __init__(self):
        self.numpy_parameter_transfers = 0
        self.numpy_gradient_transfers = 0
        self.scalar_item_calls = 0
        self.closure_calls = 0

    def as_dict(self):
        return {
            "numpy_parameter_transfers": self.numpy_parameter_transfers,
            "numpy_gradient_transfers": self.numpy_gradient_transfers,
            "scalar_item_calls": self.scalar_item_calls,
            "closure_calls": self.closure_calls,
        }


class ClosureProfiler:
    """Track closure calls and optimizer-visible host synchronization counters."""

    def __init__(self):
        self.telemetry = OptimizerTelemetry()

    def wrap(self, closure):
        def profiled_closure(*args, **kwargs):
            self.telemetry.closure_calls += 1
            return closure(*args, **kwargs)
        return profiled_closure

    def as_dict(self):
        return self.telemetry.as_dict()

def parameters_to_vector(parameters, telemetry: OptimizerTelemetry = None):
    """ concatenate parameter data into a numpy vector """
    x_list = []
    for param in parameters:
        data_flat = torch.reshape(param.data, (-1,)).numpy(force=True)
        if telemetry is not None:
            telemetry.numpy_parameter_transfers += 1
        x_list.append(data_flat)

    x = np.hstack(x_list)
    return x

def parameter_gradients(parameters, telemetry: OptimizerTelemetry = None):
    """ concatenate gradients of parameters into a numpy vector """
    grad_list = []
    for param in parameters:
        dfdx_flat = torch.reshape(param.grad, (-1,)).numpy(force=True)
        if telemetry is not None:
            telemetry.numpy_gradient_transfers += 1
        grad_list.append(dfdx_flat)

    dfdx = np.hstack(grad_list)
    return dfdx

def vector_to_parameters(x, parameters):
    """ replace parameter data with the values from the numpy array x """
    offset = 0
    for param in parameters:
        size = param.numel()
        data = torch.reshape(torch.tensor(x[offset:offset+size]), param.size())
        param.data.copy_(data)
        offset += size

class WrappedOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            method="BFGS",
            line_search_method="Wolfe",
            constraints=None, max_steplen=None,
            callback=None, maxiter=100000,
            gtol=1.0e-6, ftol=1.0e-8,
            debug=0):
        defaults = dict(
            method=method,
            line_search_method=line_search_method,
            constraints=constraints,
            max_steplen=max_steplen,
            callback=callback,
            maxiter=maxiter,
            gtol=gtol,
            ftol=ftol,
            debug=debug)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Optimizer doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self.telemetry = OptimizerTelemetry()

    def step(self, closure):

        # wrapper around pytorch module that has the interface required by
        # `minimize()`
        def objfunc(x, requires_grad=True):
            # set parameters
            vector_to_parameters(x, self._params)
            # compute the objective function and its gradients through
            # backpropagation, if required
            f = closure()
            self.telemetry.closure_calls += 1
            if requires_grad:
                f.backward()

            if requires_grad:
                dfdx = parameter_gradients(self._params, telemetry=self.telemetry)
                self.telemetry.scalar_item_calls += 1
                return f.item(), dfdx
            else:
                self.telemetry.scalar_item_calls += 1
                return f.item()

        # initial point
        x0 = parameters_to_vector(self._params, telemetry=self.telemetry)

        # run the BFGS minimization
        res = minimize(objfunc, x0, **self.defaults)

        vector_to_parameters(res.x, self._params)


class TorchLBFGSOptimizer(torch.optim.LBFGS):
    """Torch-native LBFGS wrapper with telemetry compatible with this project."""

    def __init__(self, params, maxiter=100, gtol=1.0e-6, line_search_fn="strong_wolfe", **kwargs):
        defaults = dict(max_iter=maxiter, tolerance_grad=gtol, line_search_fn=line_search_fn)
        defaults.update(kwargs)
        super().__init__(params, **defaults)
        self.telemetry = OptimizerTelemetry()

    def step(self, closure):
        def torch_closure():
            with torch.enable_grad():
                self.zero_grad()
                loss = closure()
                self.telemetry.closure_calls += 1
                loss.backward()
                return loss
        return super().step(torch_closure)
