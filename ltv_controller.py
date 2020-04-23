from pydrake.systems.framework import BasicVector_, LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem

import numpy as np

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html

# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("LTVController_")
def LTVController_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, nq, nu, converter=None):
            LeafSystem_[T].__init__(self, converter)
            self.nq = nq
            self.nu = nu

            self.DeclareVectorInputPort("q_ref", BasicVector_[T](self.nq))
            self.DeclareVectorInputPort("q", BasicVector_[T](self.nq))
            self.DeclareVectorInputPort("u_ff", BasicVector_[T](self.nu))
            self.DeclareVectorInputPort("K", BasicVector_[T](self.nq*self.nu))

            self.DeclareVectorOutputPort("u", BasicVector_[T](self.nu), self.CalcControlSignal)

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, other.nq, other.nu, converter=converter)

        def CalcControlSignal(self, context, output):
            q_ref = self.get_input_port(0).Eval(context)
            q = self.get_input_port(1).Eval(context)
            u_ff = self.get_input_port(2).Eval(context)
            K = self.get_input_port(3).Eval(context).reshape(self.nu, self.nq)
            u = u_ff - np.dot(K, q - q_ref)
            output.SetFromVector(u)

    return Impl

LTVController = LTVController_[None]  # Default instantiation