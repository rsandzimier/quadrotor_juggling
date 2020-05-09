import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.systems.scalar_conversion import TemplateSystem

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
class Visualizer(PyPlotVisualizer):

    def __init__(self, n_quadrotors = 0, n_balls=0, ax=None):
        PyPlotVisualizer.__init__(self, ax=ax)
        self.n_quadrotors = n_quadrotors
        for _ in range(self.n_quadrotors):
            self.DeclareInputPort(PortDataType.kVectorValued, 6)
        self.n_balls = n_balls
        for _ in range(self.n_balls):
            self.DeclareInputPort(PortDataType.kVectorValued, 4)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-4, 4)

        # Initialize quadrotor visualization parameters
        self.length = .25  # moment arm (meters)

        self.base = np.vstack((1.2 * self.length * np.array([1, -1, -1, 1, 1]),
                               0.025 * np.array([1, 1, -1, -1, 1])))
        self.pin = np.vstack((0.005 * np.array([1, 1, -1, -1, 1]),
                              .1 * np.array([1, 0, 0, 1, 1])))
        a = np.linspace(0, 2 * np.pi, 50)
        self.prop = np.vstack(
            (self.length / 1.5 * np.cos(a), .1 + .02 * np.sin(2 * a)))

        # yapf: disable
        self.base_fill = []
        self.left_pin_fill = []
        self.right_pin_fill = []
        self.left_prop_fill = []
        self.right_prop_fill = []

        for _ in range(self.n_quadrotors):
            self.base_fill.append(self.ax.fill(
                self.base[0, :], self.base[1, :], zorder=1, edgecolor="k",
                facecolor=[.6, .6, .6]))
            self.left_pin_fill.append(self.ax.fill(
                self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
                facecolor=[0, 0, 0]))
            self.right_pin_fill.append(self.ax.fill(
                self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
                facecolor=[0, 0, 0]))
            self.left_prop_fill.append(self.ax.fill(
                self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
                facecolor=[0, 0, 1]))
            self.right_prop_fill.append(self.ax.fill(
                self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
                facecolor=[0, 0, 1]))
        # yapf: enable

        # Initialize ball visualization parameters
        self.radius = 0.1

        a = np.linspace(0, 2 * np.pi, 50)
        self.ball = np.vstack(
            (self.radius*np.cos(a), self.radius*np.sin(a)))

        self.ball_fill = []
        for _ in range(self.n_balls):
            self.ball_fill.append(self.ax.fill(
                self.ball[0, :], self.ball[0, :], zorder=0, edgecolor="k",
                facecolor=[1, 0, 0]))

    def draw(self, context):
        # Draw quadrotors
        for i in range(self.n_quadrotors):
            x = self.EvalVectorInput(context, i).CopyToVector()
            R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                          [np.sin(x[2]), np.cos(x[2])]])

            p = np.dot(R, self.base)
            self.base_fill[i][0].get_path().vertices[:, 0] = x[0] + p[0, :]
            self.base_fill[i][0].get_path().vertices[:, 1] = x[1] + p[1, :]

            p = np.dot(R, np.vstack(
                (-self.length + self.pin[0, :], self.pin[1, :])))
            self.left_pin_fill[i][0].get_path().vertices[:, 0] = x[0] + p[0, :]
            self.left_pin_fill[i][0].get_path().vertices[:, 1] = x[1] + p[1, :]
            p = np.dot(R, np.vstack((self.length + self.pin[0, :], self.pin[1, :])))
            self.right_pin_fill[i][0].get_path().vertices[:, 0] = x[0] + p[0, :]
            self.right_pin_fill[i][0].get_path().vertices[:, 1] = x[1] + p[1, :]

            p = np.dot(R,
                       np.vstack((-self.length + self.prop[0, :], self.prop[1, :])))
            self.left_prop_fill[i][0].get_path().vertices[:, 0] = x[0] + p[0, :]
            self.left_prop_fill[i][0].get_path().vertices[:, 1] = x[1] + p[1, :]

            p = np.dot(R, np.vstack(
                (self.length + self.prop[0, :], self.prop[1, :])))
            self.right_prop_fill[i][0].get_path().vertices[:, 0] = x[0] + p[0, :]
            self.right_prop_fill[i][0].get_path().vertices[:, 1] = x[1] + p[1, :]
        # Draw balls
        for i in range(self.n_balls):
            x = self.EvalVectorInput(context, i + self.n_quadrotors).CopyToVector()
            R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                          [np.sin(x[2]), np.cos(x[2])]])

            self.ball_fill[i][0].get_path().vertices[:, 0] = x[0] + self.ball[0, :]
            self.ball_fill[i][0].get_path().vertices[:, 1] = x[1] + self.ball[1, :]

        self.ax.set_title("t = {:.1f}".format(context.get_time()))
