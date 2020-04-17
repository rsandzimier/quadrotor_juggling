import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
from pydrake.systems.scalar_conversion import TemplateSystem

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("Quadrotor2D_")
def Quadrotor2D_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, n_quadrotors=0, n_balls=0, converter=None):
            # n_quadrotors is the number of OTHER quadrotors (besides this one)
            LeafSystem_[T].__init__(self, converter)
            self.n_quadrotors = n_quadrotors
            self.n_balls = n_balls
            # two inputs (thrust)
            self.DeclareVectorInputPort("u", BasicVector_[T](2))

            # six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6),
                                         self.CopyStateOut)
            # three positions, three velocities
            self.DeclareContinuousState(3, 3, 0)

            # Other quadrotor states as inputs
            max_quads = 5
            # NOTE: For some reason, using a loop to call self.DeclareVectorInputPort n_quadrotors number
            # of times leads to a segmentation fault. As a dirty fix, declare some hard-coded maximum number
            # of vector inputs instead of using a for loop
            assert n_quadrotors <= max_quads, "Max number of quadrotors exceeded in Quadrotor2D."
            self.DeclareVectorInputPort("quad_0", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_1", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_2", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_3", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_4", BasicVector_[T](6))

            # Ball states as inputs
            max_balls = 5
            # NOTE: For some reason, using a loop to call self.DeclareVectorInputPort n_balls number
            # of times leads to a segmentation fault. As a dirty fix, declare some hard-coded maximum number
            # of vector inputs instead of using a for loop
            assert n_balls <= max_balls, "Max number of balls exceeded in Quadrotor2D."
            self.DeclareVectorInputPort("ball_0", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_1", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_2", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_3", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_4", BasicVector_[T](4))

            # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
            self.length = 0.25  # length of rotor arm
            self.mass = 0.486  # mass of quadrotor
            self.inertia = 0.00383  # moment of inertia
            self.gravity = 9.81  # gravity

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()
            q = x[:3]
            qdot = x[3:]
            qddot = np.array([
                -np.sin(q[2]) / self.mass * (u[0] + u[1]),
                np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])
            ])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

    return Impl


Quadrotor2D = Quadrotor2D_[None]  # Default instantiation
