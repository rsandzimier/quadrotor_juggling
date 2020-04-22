import numpy as np

# TO DO: Allow for variable radius balls
r_ball = 0.1
w_quad = 0.3 # Half-length
h_quad = 0.025 # Half-height

def CalcSignedInterferenceBallBall(q_ball1, q_ball2):
    q_diff = q_ball1 - q_ball2
    dist = np.linalg.norm(q_diff)
    signed_interference = np.zeros(2)
    if dist > 0 and dist <= 2*r_ball:
        signed_interference = 0.5 * q_diff * (2*r_ball - dist)/dist
    return signed_interference

def CalcSignedInterferenceBallQuad(q_ball, q_quad):
    return CalcSignedInterferenceBallQuadAux(q_ball, q_quad, True)

def CalcSignedInterferenceQuadBall(q_quad, q_ball):
    return CalcSignedInterferenceBallQuadAux(q_ball, q_quad, False)

def CalcSignedInterferenceBallQuadAux(q_ball, q_quad, ball_to_quad):

    # Calc closest point on quad to center of ball
    p_closest = CalcClosestLocationQuadBall(q_quad, q_ball)
    q_diff = q_ball - p_closest
    q_diff_norm = np.linalg.norm(q_diff)
    # if not ball_to_quad:
        # print(q_ball, q_quad, p_closest, q_diff_norm)

    q_diff_unit = q_diff / q_diff_norm if q_diff_norm > 0 else q_diff
    r = q_diff_unit*r_ball
    if np.dot(r - q_quad[:2], q_ball-q_quad[:2]) < 0:
        r = -r
    signed_interference = r - q_diff
    if np.dot(signed_interference, q_ball-q_quad[:2]) < 0:
        signed_interference = signed_interference*0
    if q_diff_norm > r_ball:
        signed_interference = signed_interference*0

    if not ball_to_quad:
        signed_interference = -signed_interference
        # print (signed_interference)
    return signed_interference

def CalcSignedInterferenceQuadQuad(q_quad1, q_quad2):

    pass

def CalcClosestLocationQuadBall(q_quad, q_ball):
    R_i = np.array([[np.cos(q_quad[2]) , -np.sin(q_quad[2])],
                    [np.sin(q_quad[2]) ,  np.cos(q_quad[2])]])
    R_inv_i = R_i.T

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

def CalcClosestLocationQuadQuad(q_quad1, q_quad2):
    pass

def CalcDampedVelocityBall(qdot_diff, signed_interference):
    signed_interference_norm = np.linalg.norm(signed_interference)
    signed_interference_unit = signed_interference/signed_interference_norm if signed_interference_norm > 0 else np.zeros(2)
    return np.dot(qdot_diff, signed_interference_unit)*signed_interference_unit
