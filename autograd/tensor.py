from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.array] # https://docs.python.org/3/library/typing.html#typing.Callable a function that takes np.ndarray and returns np.array

#will allow argument to be a float, list, or np.ndarray when given to function or class?
Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    """
    Ensures that an "arrayable" object is type np.ndarray and if it is not then it returns an
    np.ndarray of the "arrayable

    Parameter:
        arrayable: a float, list or np.ndarray as specified by Arrayable

    Returns:
        np.ndarray
    """
    #https://www.w3schools.com/python/ref_func_isinstance.asp
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)
    
class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad

    def zero_grad(self) -> None:
        #returns nothing... zeros out gradient when called
        self.grad = Tensor(np.zeros_like(self.data))

    # https://docs.python.org/3/library/functions.html#repr returns string representation of object
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def backwards(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
            
        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backwards(Tensor(backward_grad))
        
        def sum(self) -> 'Tensor':
            return tensor_sum(self)
        
        def tensor_sum(t: Tensor) -> Tensor:
            """
            Takes a tensor and returns the 0-tensor (scalars have 0 rank?)
            that's the sum of all its elements
            """
            data = t.data.sum()
            requires_grad = t.requires_grad
            if requires_grad:
                def grad_fn(grad: np.ndarray) -> np.array:
                    """
                    grad is a 0-tensor, so each input element contributes that much????

                    t = [5,6,7,8]
                    t2 = t.sum() #26

                    adding any amount to any one of the elements of t adds the amount to the sum t2
                    """
                    return grad * np.ones_like(t.data)
            
            depends_on = []

            return Tensor(data,
                          requires_grad,
                          depends_on)