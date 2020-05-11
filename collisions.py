import numpy as np
from pydrake.symbolic import Variable, cos, sin

# TO DO: Allow for variable radius/mass balls
r_ball = 0.1
m_ball = 0.1
w_quad = 0.3 # Half-length
h_quad = 0.025 # Half-height
m_quad = 0.486
I_quad = 0.00383

def CalcClosestDistanceBallBall(q_ball1, q_ball2):
    q_diff = q_ball1 - q_ball2
    dist = np.linalg.norm(q_diff)
    return dist - 2*r_ball

def CalcClosestDistanceQuadBall(q_quad, q_ball):
    p_closest = CalcClosestLocationQuadBall(q_quad, q_ball)
    q_diff = q_ball - p_closest
    dist = np.linalg.norm(q_diff)
    return dist - r_ball

def CalcPostCollisionStateBallBall(x_ball, x_ball_other):
    # Calculate state of ball with initial state x after colliding with ball of state x_other
    # This function is meant to only be called once a witness function verifies the balls are in collision
    q_par = x_ball_other[0:2] - x_ball[0:2]
    q_par_norm = np.linalg.norm(q_par)
    q_par_unit = q_par / q_par_norm if q_par_norm > 0 else np.array([0,1]) 
    q_perp_unit = np.array([q_par_unit[1], -q_par_unit[0]])

    dist = CalcClosestDistanceBallBall(x_ball[0:2], x_ball_other[0:2])

    v_par = np.dot(x_ball[2:4], q_par_unit)
    v_perp = np.dot(x_ball[2:4], q_perp_unit)
    v_par_other = np.dot(x_ball_other[2:4], q_par_unit)

    m = m_ball
    m_other = m_ball

    v_par_new = ((m - m_other)*v_par + 2*m_other*v_par_other)/(m + m_other)

    x_ball[0:2] = x_ball[0:2] + dist*q_par_unit
    x_ball[2:4] = v_par_new*q_par_unit + v_perp*q_perp_unit

    return x_ball

def CalcPostCollisionStateQuadBall(x_quad, x_ball):
    return CalcPostCollisionStateQuadBallAux(x_quad, x_ball, True)

def CalcPostCollisionStateBallQuad(x_ball, x_quad):
    return CalcPostCollisionStateQuadBallAux(x_quad, x_ball, False)

def CalcPostCollisionStateQuadBallAux(x_quad, x_ball, return_quad_state):
    # Calculate state of quad (or ball) with initial state x_quad (or x_ball) after colliding
    # with ball (or quad) of state x_ball (or x_quad). (parentheses refer to when return_quad_state is False)
    # This function is meant to only be called once a witness function verifies the quad/ball are in collision

    # Calc closest point on quad to center of ball
    p_closest = CalcClosestLocationQuadBall(x_quad[0:3], x_ball[0:2])
    q_par = p_closest - x_ball[0:2]
    q_par_norm = np.linalg.norm(q_par)
    q_par_unit = q_par / q_par_norm if q_par_norm > 0 else np.array([0,1]) 
    q_perp_unit = np.array([q_par_unit[1], -q_par_unit[0]])

    dist = CalcClosestDistanceQuadBall(x_quad[0:3], x_ball[0:2])

    # v1, v2, w, r1, r2 
    v_quad_par = np.dot(x_quad[3:5], q_par_unit)
    v_quad_perp = np.dot(x_quad[3:5], q_perp_unit)
    w_quad = x_quad[5]
    v_ball_par = np.dot(x_ball[2:4], q_par_unit)
    v_ball_perp = np.dot(x_ball[2:4], q_perp_unit)

    p_com = (x_quad[0:2]*m_quad + x_ball[0:2]*m_ball)/(m_quad + m_ball)

    moment_arm_quad = np.dot(x_quad[0:2] - p_com, q_perp_unit) # Moment arm about center of mass of system (quad + ball)
    moment_arm_ball = np.dot(x_ball[0:2] - p_com, q_perp_unit) # Moment arm about center of mass of system (quad + ball)

    v_quad_par_new = (m_quad*m_ball*I_quad*(v_quad_par*(m_quad - m_ball) + 2*m_ball*v_ball_par) + \
                            m_quad**2*m_ball**2*(moment_arm_quad - moment_arm_ball)**2*v_quad_par) / \
                            (m_quad*m_ball*(m_quad + m_ball)*I_quad + \
                            m_quad**2*m_ball**2*(moment_arm_quad - moment_arm_ball)**2)

    v_ball_par_new = (m_quad*m_ball*I_quad*(v_ball_par*(m_ball - m_quad) + 2*m_quad*v_quad_par) + \
                        m_quad**2*m_ball**2*(moment_arm_quad - moment_arm_ball)**2*v_ball_par) / \
                        (m_quad*m_ball*(m_quad + m_ball)*I_quad + \
                        m_quad**2*m_ball**2*(moment_arm_quad - moment_arm_ball)**2)

    w_quad_new = w_quad + (m_quad*moment_arm_quad*(v_quad_par-v_quad_par_new) + m_ball*moment_arm_ball*(v_ball_par-v_ball_par_new))/I_quad

    if return_quad_state:
        x_quad[0:2] = x_quad[0:2] + dist*q_par_unit # Right now, only x,y position of quad update. In future, could look into updating theta too
        x_quad[3:5] = v_quad_par_new*q_par_unit + v_quad_perp*q_perp_unit
        x_quad[5] = w_quad_new
        return x_quad
    else: # return ball state
        x_ball[0:2] = x_ball[0:2] - dist*q_par_unit
        x_ball[2:4] = v_ball_par_new*q_par_unit + v_ball_perp*q_perp_unit
        return x_ball

def CalcClosestLocationQuadBall(q_quad, q_ball):
    if not isinstance(q_quad[2], Variable):
        R_i = np.array([[np.cos(q_quad[2]) , -np.sin(q_quad[2])],
                    [np.sin(q_quad[2]) ,  np.cos(q_quad[2])]])
    else:
        R_i = np.array([[cos(q_quad[2]) , -sin(q_quad[2])],
            [sin(q_quad[2]) ,  cos(q_quad[2])]])

    R_inv_i = R_i.T
# 
    q_dash = R_inv_i.dot(q_ball - q_quad[:2])

    p_closest = np.clip(q_dash, [-w_quad, -h_quad], [w_quad, h_quad])    

    # Handle case where q_ball is inside the quadrotor base
    if p_closest[0] > -w_quad and p_closest[0] < w_quad and p_closest[1] > -h_quad and p_closest[1] < h_quad:
        if p_closest[1] >= 0 and p_closest[1] >= h_quad -w_quad - p_closest[0] and p_closest[1] >= h_quad -w_quad + p_closest[0]:
            p_closest[1] = h_quad
        elif p_closest[1] < 0 and p_closest[1] <= -h_quad -w_quad + p_closest[0] and p_closest[1] <= -h_quad -w_quad - p_closest[0]:
            p_closest[1] = -h_quad
        elif p_closest[0] < -w_quad+h_quad:
            p_closest[0] = -w_quad
        else: # p_closest[0] > w_quad-h_quad
            p_closest[0] = w_quad
    return q_quad[:2] + R_i.dot(p_closest)

def CalcPostCollisionStateQuadBallResidual(vars):
    x_quad = vars[0:6]
    x_ball =vars[6:10]
    x_ref = vars[10:16]
    return CalcPostCollisionStateQuadBallAux(x_quad, x_ball, True) - x_ref

def CalcPostCollisionStateBallQuadResidual(vars):
    x_quad = vars[0:6]
    x_ball =vars[6:10]
    x_ref = vars[10:14]
    return CalcPostCollisionStateQuadBallAux(x_quad, x_ball, False) - x_ref