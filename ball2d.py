import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.symbolic import Expression
import time

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
            self.mass = 1.0
            self.gravity = 9.81*0.0
            self.mu = 0.3
            
            self.stiffness_ball = 1000.0
            self.damping_ball = 0.0

            self.width_quad = 0.25

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

            print(np.linalg.norm(qdot))

            if not isinstance(q[0], Expression): 

                # calculate collisions with quadrotos
                for i in range(self.n_quadrotors):
                    x_quad_i = self.EvalVectorInput(context,i).CopyToVector()
                    q_quad_i = x_quad_i[:3]
                    qdot_quad_i = x_quad_i[3:]
                    R_i = np.array([[np.cos(q_quad_i[2]) , -np.sin(q_quad_i[2])],
                                    [np.sin(q_quad_i[2]) ,  np.cos(q_quad_i[2])]])

                    q_dash = R_i.dot(q - q_quad_i[:2])
                    qdot_dash = R_i.dot(qdot - qdot_quad_i[:2])

                    if ((q_dash[0] > -self.width_quad ) and (q_dash[0] < self.width_quad ) and (q_dash[1] > 0) and (q_dash[1] < self.radius)):

                        R_inv_i = np.array([[np.cos(-q_quad_i[2]) , -np.sin(-q_quad_i[2])],
                                            [np.sin(-q_quad_i[2]) ,  np.cos(-q_quad_i[2])]])

                        # add elastic force
                        f += R_inv_i.dot((self.stiffness_ball) * np.array([0.0, self.radius-q_dash[1]]))
                        # f += R_inv_i.dot((self.stiffness_ball) * np.array([-np.sign(qdot_dash[0]).astype(float)*(self.radius-q_dash[1]), self.radius-q_dash[1]]))

                        # add damping force
                        f += R_inv_i.dot((self.damping_ball) * np.array([0.0, -qdot_dash[1]]))

                # calculate forces from collision with other balls
                for i in range(self.n_quadrotors,self.n_quadrotors + self.n_balls):
                    x_ball_i = self.EvalVectorInput(context,i).CopyToVector()
                    q_ball_i = x_ball_i[:2]
                    qdot_ball_i = x_ball_i[2:]

                    Delta_q = q - q_ball_i
                    Delta_qdot = qdot - qdot_ball_i

                    dist = np.linalg.norm(Delta_q)

                    if dist <= 2*self.radius:
                        # add elastic force
                        f += (0.5*self.stiffness_ball) * Delta_q * (2*self.radius - dist)/dist
                        # add damping force
                        f += (0.5*self.damping_ball) * -1.0* Delta_qdot 

            qddot = f + np.array([0,-self.gravity])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

    return Impl


Ball2D = Ball2D_[None]  # Default instantiation