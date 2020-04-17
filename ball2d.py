import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.symbolic import Expression

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("Ball2D_")
def Ball2D_(T):

    class Impl(LeafSystem_[T]):
        def _construct(self, n_quadrotors=0, n_balls=0, converter=None):
            # n_balls is the number of OTHER balls (besides this one)

            LeafSystem_[T].__init__(self, converter)
            self.n_quadrotors = n_quadrotors
            self.n_balls = n_balls
            # four outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](4),
                                         self.CopyStateOut)
            # two positions, two velocities
            self.DeclareContinuousState(2, 2, 0)

            # Quadrotor states as inputs
            max_quads = 5
            # NOTE: For some reason, using a loop to call self.DeclareVectorInputPort n_quadrotors number
            # of times leads to a segmentation fault. As a dirty fix, declare some hard-coded maximum number
            # of vector inputs instead of using a for loop
            assert n_quadrotors <= max_quads, "Max number of quadrotors exceeded in Ball2D."
            self.DeclareVectorInputPort("quad_0", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_1", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_2", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_3", BasicVector_[T](6))
            self.DeclareVectorInputPort("quad_4", BasicVector_[T](6))

            # Other ball states as inputs
            max_balls = 5
            # NOTE: For some reason, using a loop to call self.DeclareVectorInputPort n_balls number
            # of times leads to a segmentation fault. As a dirty fix, declare some hard-coded maximum number
            # of vector inputs instead of using a for loop
            assert n_balls <= max_balls, "Max number of balls exceeded in Ball2D."
            self.DeclareVectorInputPort("ball_0", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_1", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_2", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_3", BasicVector_[T](4))
            self.DeclareVectorInputPort("ball_4", BasicVector_[T](4))

            self.radius = 0.1
            self.mass = 1.0
            self.gravity = 9.81

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            q = x[:2]
            qdot = x[2:]

            qddot = np.array([0,-self.gravity])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

    return Impl


Ball2D = Ball2D_[None]  # Default instantiation