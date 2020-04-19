import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.symbolic import Expression
import time
from collisions import CalcSignedInterferenceBallBall, CalcSignedInterferenceBallQuad, CalcDampedVelocityBall

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
            for i in range(self.n_quadrotors):
                    self.DeclareVectorInputPort("quad_" + str(i), BasicVector_[T](6))

            # Other ball states as inputs
            for i in range(self.n_balls):
                self.DeclareVectorInputPort("ball_" + str(i), BasicVector_[T](4))
            self.radius = 0.1
            self.mass = 0.1
            self.gravity = 9.81
            self.mu = 0.3
            
            self.stiffness_ball = 10000.0
            self.damping_ball = 0.0

            self.width_quad = 0.3

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, n_quadrotors=other.n_quadrotors, n_balls=other.n_balls, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            q = x[:2]
            qdot = x[2:]

            f = np.array([0.0,0.0])

            # print(0.5*self.mass*np.linalg.norm(qdot)**2 + self.mass*self.gravity*q[1])

            if not isinstance(q[0], Expression): 

                # calculate collisions with quadrotors
                for i in range(self.n_quadrotors):
                    x_quad_i = self.EvalVectorInput(context,i).CopyToVector()
                    q_quad_i = x_quad_i[:3]
                    qdot_quad_i = x_quad_i[3:]

                    signed_interference = CalcSignedInterferenceBallQuad(q, q_quad_i)
                    qdot_diff = qdot - qdot_quad_i[:2]
                    qdot_damped = CalcDampedVelocityBall(qdot_diff, signed_interference) # Only damp the component of velocity parallel to the interference

                    # add elastic force
                    f += self.stiffness_ball * signed_interference
                    # add damping force
                    f += -self.damping_ball*qdot_damped

                # calculate forces from collision with other balls
                for i in range(self.n_quadrotors,self.n_quadrotors + self.n_balls):
                    x_ball_i = self.EvalVectorInput(context,i).CopyToVector()
                    q_ball_i = x_ball_i[:2]
                    qdot_ball_i = x_ball_i[2:]
                    signed_interference = CalcSignedInterferenceBallBall(q, q_ball_i)

                    qdot_diff = qdot - qdot_ball_i
                    qdot_damped = CalcDampedVelocityBall(qdot_diff, signed_interference) # Only damp the component of velocity parallel to the interference


                    # add elastic force
                    f += self.stiffness_ball*signed_interference
                    # add damping force
                    f += -self.damping_ball*qdot_damped

            qddot = f/self.mass + np.array([0,-self.gravity])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

    return Impl


Ball2D = Ball2D_[None]  # Default instantiation