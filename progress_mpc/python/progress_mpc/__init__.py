"""Progress-maximizing MPC (acados SQP-RTI) in C++, with Python bindings.

`ProgressMpc` is the plain MPC. `Ac4Mpc`, the actor- and critic-guided one (AC4MPC-RTI), is there when the package was
built with libtorch and an export of its OCP; otherwise it is None.
"""

from ._core import ProgressMpc

try:
    from ._ac4mpc import Ac4Mpc
except ImportError:  # built without libtorch or without generated_ac4mpc/
    Ac4Mpc = None

__all__ = ["Ac4Mpc", "ProgressMpc"]
