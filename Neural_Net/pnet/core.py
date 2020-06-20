__all__ = [
    "enable_eager_execution", "disable_eager_execution", "executing_eagerly"
]

import sys
this = sys.modules[__name__]
this._eager = True
this._requires_grad = True

def enable_eager_execution():
    this._eager = True

def disable_eager_execution():
    this._eager = False

def executing_eagerly():
    return this._eager

def enable_gradient():
    this._requires_grad = True

def disable_gradient():
    this._requires_grad = False

def requiring_gradient():
    return this._requires_grad