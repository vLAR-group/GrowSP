import numpy as np
from scipy.linalg import expm, norm

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class trans_coords:
    def __init__(self, shift_ratio):
        self.ratio = shift_ratio

    def __call__(self, coords):
        shift = (np.random.uniform(0, 1, 3) * self.ratio)
        return coords + shift


class rota_coords:
    def __init__(self, rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi))):
        self.rotation_bound = rotation_bound

    def __call__(self, coords):
        rot_mats = []
        for axis_ind, rot_bound in enumerate(self.rotation_bound):
            theta = 0
            axis = np.zeros(3)
            axis[axis_ind] = 1
            if rot_bound is not None:
                theta = np.random.uniform(*rot_bound)
            rot_mats.append(M(axis, theta))
        # Use random order
        np.random.shuffle(rot_mats)
        rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
        return coords.dot(rot_mat)


class scale_coords:
    def __init__(self, scale_bound=(0.8, 1.25)):
        self.scale_bound = scale_bound

    def __call__(self, coords):
        scale = np.random.uniform(*self.scale_bound)
        return coords*scale