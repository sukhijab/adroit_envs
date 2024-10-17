from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import error

try:
    import mujoco
    from mujoco import MjData, MjModel, mjtObj
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco")

MJ_OBJ_TYPES = [
    "mjOBJ_BODY",
    "mjOBJ_JOINT",
    "mjOBJ_GEOM",
    "mjOBJ_SITE",
    "mjOBJ_CAMERA",
    "mjOBJ_ACTUATOR",
    "mjOBJ_SENSOR",
]


def robot_get_obs(model, data, joint_names):
    """Returns all joint positions and velocities associated with a robot."""
    if data.qpos is not None and joint_names:
        names = [n for n in joint_names if n.startswith("robot")]
        return (
            np.squeeze(np.array([get_joint_qpos(model, data, name) for name in names])),
            np.squeeze(np.array([get_joint_qvel(model, data, name) for name in names])),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(model, data, action):
    """For torque actuators it copies the action into mujoco ctrl field.

    For position actuators it sets the target relative to the current qpos.
    """
    if model.nmocap > 0:
        _, action = np.split(action, (model.nmocap * 7,))

    if len(data.ctrl) > 0:
        for i in range(action.shape[0]):
            if model.actuator_biastype[i] == 0:
                data.ctrl[i] = action[i]
            else:
                idx = model.jnt_qposadr[model.actuator_trnid[i, 0]]
                data.ctrl[i] = data.qpos[idx] + action[i]


def mocap_set_action(model, data, action):
    """Update the position of the mocap body with the desired action.

    The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if model.nmocap > 0:
        action, _ = np.split(action, (model.nmocap * 7,))
        action = action.reshape(model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(model, data)
        data.mocap_pos[:] = data.mocap_pos + pos_delta
        data.mocap_quat[:] = data.mocap_quat + quat_delta


def reset_mocap_welds(model, data):
    """Resets the mocap welds that we use for actuation."""
    if model.nmocap > 0 and model.eq_data is not None:
        for i in range(model.eq_data.shape[0]):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                model.eq_data[i, :7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mujoco.mj_forward(model, data)


def reset_mocap2body_xpos(model, data):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if model.eq_type is None or model.eq_obj1id is None or model.eq_obj2id is None:
        return
    for eq_type, obj1_id, obj2_id in zip(
        model.eq_type, model.eq_obj1id, model.eq_obj2id
    ):
        if eq_type != mujoco.mjtEq.mjEQ_WELD:
            continue

        mocap_id = model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        data.mocap_pos[mocap_id][:] = data.xpos[body_idx]
        data.mocap_quat[mocap_id][:] = data.xquat[body_idx]


def get_site_jacp(model, data, site_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding site id.
    """
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    return jacp


def get_site_jacr(model, data, site_id):
    """Return the Jacobian' rotational component of the end-effector of
    the corresponding site id.
    """
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, None, jacr, site_id)

    return jacr


def set_joint_qpos(model, data, name, value):
    """Set the joint positions (qpos) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), f"Value has incorrect shape {name}: {value}"
    data.qpos[start_idx:end_idx] = value


def set_joint_qvel(model, data, name, value):
    """Set the joints linear and angular (qvel) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 3
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), f"Value has incorrect shape {name}: {value}"
    data.qvel[start_idx:end_idx] = value


def get_joint_qpos(model, data, name):
    """Return the joints position and orientation (qpos) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qpos[start_idx:end_idx].copy()


def get_joint_qvel(model, data, name):
    """Return the joints linear and angular velocities (qvel) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qvel[start_idx:end_idx].copy()


def get_site_xpos(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert site_id != -1, f"Site with name '{name}' is not part of the model!"
    return data.site_xpos[site_id]


def get_site_xvelp(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert site_id != -1, f"Site with name '{name}' is not part of the model!"
    jacp = get_site_jacp(model, data, site_id)
    xvelp = jacp @ data.qvel
    return xvelp


def get_site_xvelr(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert site_id != -1, f"Site with name '{name}' is not part of the model!"
    jacp = get_site_jacr(model, data, site_id)
    xvelp = jacp @ data.qvel
    return xvelp


def set_mocap_pos(model, data, name, value):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id != -1, f"Body with name '{name}' is not part of the model!"
    mocap_id = model.body_mocapid[body_id]
    data.mocap_pos[mocap_id] = value


def set_mocap_quat(model: MjModel, data: MjData, name: str, value):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id != -1, f"Body with name '{name}' is not part of the model!"
    mocap_id = model.body_mocapid[body_id]
    data.mocap_quat[mocap_id] = value


def get_site_xmat(model: MjModel, data: MjData, name: str):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert site_id != -1, f"Site with name '{name}' is not part of the model!"
    return data.site_xmat[site_id].reshape(3, 3)


def extract_mj_names(
    model: MjModel, obj_type: mjtObj
) -> Tuple[Union[Tuple[str, ...], Tuple[()]], Dict[str, int], Dict[int, str]]:

    if obj_type == mujoco.mjtObj.mjOBJ_BODY:
        name_addr = model.name_bodyadr
        n_obj = model.nbody

    elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        name_addr = model.name_jntadr
        n_obj = model.njnt

    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        name_addr = model.name_geomadr
        n_obj = model.ngeom

    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        name_addr = model.name_siteadr
        n_obj = model.nsite

    elif obj_type == mujoco.mjtObj.mjOBJ_LIGHT:
        name_addr = model.name_lightadr
        n_obj = model.nlight

    elif obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
        name_addr = model.name_camadr
        n_obj = model.ncam

    elif obj_type == mujoco.mjtObj.mjOBJ_ACTUATOR:
        name_addr = model.name_actuatoradr
        n_obj = model.nu

    elif obj_type == mujoco.mjtObj.mjOBJ_SENSOR:
        name_addr = model.name_sensoradr
        n_obj = model.nsensor

    elif obj_type == mujoco.mjtObj.mjOBJ_TENDON:
        name_addr = model.name_tendonadr
        n_obj = model.ntendon

    elif obj_type == mujoco.mjtObj.mjOBJ_MESH:
        name_addr = model.name_meshadr
        n_obj = model.nmesh
    else:
        raise ValueError(
            "`{}` was passed as the MuJoCo model object type. The MuJoCo model object type can only be of the following mjtObj enum types: {}.".format(
                obj_type, MJ_OBJ_TYPES
            )
        )

    id2name = {i: None for i in range(n_obj)}
    name2id = {}
    for addr in name_addr:
        name = model.names[addr:].split(b"\x00")[0].decode()
        if name:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name

    return tuple(id2name[id] for id in sorted(name2id.values())), name2id, id2name


class MujocoModelNames:
    """Access mjtObj object names and ids of the current MuJoCo model.

    This class supports access to the names and ids of the following mjObj types:
        mjOBJ_BODY
        mjOBJ_JOINT
        mjOBJ_GEOM
        mjOBJ_SITE
        mjOBJ_CAMERA
        mjOBJ_ACTUATOR
        mjOBJ_SENSOR

    The properties provided for each ``mjObj`` are:
        ``mjObj``_names: list of the mjObj names in the model of type mjOBJ_FOO.
        ``mjObj``_name2id: dictionary with name of the mjObj as keys and id of the mjObj as values.
        ``mjObj``_id2name: dictionary with id of the mjObj as keys and name of the mjObj as values.
    """

    def __init__(self, model: MjModel):
        """Access mjtObj object names and ids of the current MuJoCo model.

        Args:
            model: mjModel of the MuJoCo environment.
        """
        (
            self._body_names,
            self._body_name2id,
            self._body_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_BODY)
        (
            self._joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_JOINT)
        (
            self._geom_names,
            self._geom_name2id,
            self._geom_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_GEOM)
        (
            self._site_names,
            self._site_name2id,
            self._site_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SITE)
        (
            self._camera_names,
            self._camera_name2id,
            self._camera_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_CAMERA)
        (
            self._actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
        (
            self._sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SENSOR)

    @property
    def body_names(self):
        return self._body_names

    @property
    def body_name2id(self):
        return self._body_name2id

    @property
    def body_id2name(self):
        return self._body_id2name

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def joint_name2id(self):
        return self._joint_name2id

    @property
    def joint_id2name(self):
        return self._joint_id2name

    @property
    def geom_names(self):
        return self._geom_names

    @property
    def geom_name2id(self):
        return self._geom_name2id

    @property
    def geom_id2name(self):
        return self._geom_id2name

    @property
    def site_names(self):
        return self._site_names

    @property
    def site_name2id(self):
        return self._site_name2id

    @property
    def site_id2name(self):
        return self._site_id2name

    @property
    def camera_names(self):
        return self._camera_names

    @property
    def camera_name2id(self):
        return self._camera_name2id

    @property
    def camera_id2name(self):
        return self._camera_id2name

    @property
    def actuator_names(self):
        return self._actuator_names

    @property
    def actuator_name2id(self):
        return self._actuator_name2id

    @property
    def actuator_id2name(self):
        return self._actuator_id2name

    @property
    def sensor_names(self):
        return self._sensor_names

    @property
    def sensor_name2id(self):
        return self._sensor_name2id

    @property
    def sensor_id2name(self):
        return self._sensor_id2name

# Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Many methods borrow heavily or entirely from transforms3d:
# https://github.com/matthew-brett/transforms3d
# They have mostly been modified to support batched operations.

import itertools
import math

import numpy as np

"""
Rotations
=========

Note: these have caused many subtle bugs in the past.
Be careful while updating these methods and while using them in clever ways.

See MuJoCo documentation here: http://mujoco.org/book/modeling.html#COrientation

Conventions
-----------
    - All functions accept batches as well as individual rotations
    - All rotation conventions match respective MuJoCo defaults
    - All angles are in radians
    - Matrices follow LR convention
    - Euler Angles are all relative with 'xyz' axes ordering
    - See specific representation for more information

Representations
---------------

Euler
    There are many euler angle frames -- here we will strive to use the default
        in MuJoCo, which is eulerseq='xyz'.
    This frame is a relative rotating frame, about x, y, and z axes in order.
        Relative rotating means that after we rotate about x, then we use the
        new (rotated) y, and the same for z.

Quaternions
    These are defined in terms of rotation (angle) about a unit vector (x, y, z)
    We use the following <q0, q1, q2, q3> convention:
            q0 = cos(angle / 2)
            q1 = sin(angle / 2) * x
            q2 = sin(angle / 2) * y
            q3 = sin(angle / 2) * z
        This is also sometimes called qw, qx, qy, qz.
    Note that quaternions are ambiguous, because we can represent a rotation by
        angle about vector <x, y, z> and -angle about vector <-x, -y, -z>.
        To choose between these, we pick "first nonzero positive", where we
        make the first nonzero element of the quaternion positive.
    This can result in mismatches if you're converting an quaternion that is not
        "first nonzero positive" to a different representation and back.

Axis Angle
    (Not currently implemented)
    These are very straightforward.  Rotation is angle about a unit vector.

XY Axes
    (Not currently implemented)
    We are given x axis and y axis, and z axis is cross product of x and y.

Z Axis
    This is NOT RECOMMENDED.  Defines a unit vector for the Z axis,
        but rotation about this axis is not well defined.
    Instead pick a fixed reference direction for another axis (e.g. X)
        and calculate the other (e.g. Y = Z cross-product X),
        then use XY Axes rotation instead.

SO3
    (Not currently implemented)
    While not supported by MuJoCo, this representation has a lot of nice features.
    We expect to add support for these in the future.

TODO / Missing
--------------
    - Rotation integration or derivatives (e.g. velocity conversions)
    - More representations (SO3, etc)
    - Random sampling (e.g. sample uniform random rotation)
    - Performance benchmarks/measurements
    - (Maybe) define everything as to/from matrices, for simplicity
"""

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def euler2mat(euler):
    """Convert Euler Angles to Rotation Matrix.

    See rotation.py for notes
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shaped euler {euler}"

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """Convert Euler Angles to Quaternions.

    See rotation.py for notes
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """Convert Rotation Matrix to Euler Angles.

    See rotation.py for notes
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat):
    """Convert Rotation Matrix to Quaternion.

    See rotation.py for notes
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """Convert Quaternion to Euler Angles.

    See rotation.py for notes
    """
    return mat2euler(quat2mat(quat))


def subtract_euler(e1, e2):

    assert e1.shape == e2.shape
    assert e1.shape[-1] == 3
    q1 = euler2quat(e1)
    q2 = euler2quat(e2)
    q_diff = quat_mul(q1, quat_conjugate(q2))
    return quat2euler(q_diff)


def quat2mat(quat):
    """Convert Quaternion to Euler Angles.

    See rotation.py for notes
    """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, f"Invalid shape quat {quat}"

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1):
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_rot_vec(q, v0):
    q_v0 = np.array([0, v0[0], v0[1], v0[2]])
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[1:]
    return v


def quat_identity():
    return np.array([1, 0, 0, 0])


def quat2axisangle(quat):
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def euler2point_euler(euler):
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 3
    _euler_sin = np.sin(_euler)
    _euler_cos = np.cos(_euler)
    return np.concatenate([_euler_sin, _euler_cos], axis=-1)


def point_euler2euler(euler):
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 6
    angle = np.arctan(_euler[..., :3] / _euler[..., 3:])
    angle[_euler[..., 3:] < 0] += np.pi
    return angle


def quat2point_quat(quat):
    # Should be in qw, qx, qy, qz
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 4
    angle = np.arccos(_quat[:, [0]]) * 2
    xyz = _quat[:, 1:]
    xyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (xyz / np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([np.sin(angle), np.cos(angle), xyz], axis=-1)


def point_quat2quat(quat):
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 5
    angle = np.arctan(_quat[:, [0]] / _quat[:, [1]])
    qw = np.cos(angle / 2)

    qxyz = _quat[:, 2:]
    qxyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (qxyz * np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([qw, qxyz], axis=-1)


def normalize_angles(angles):
    """Puts angles in [-pi, pi] range."""
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -np.pi - 1e-6 <= angles.min() and angles.max() <= np.pi + 1e-6
    return angles


def round_to_straight_angles(angles):
    """Returns closest angle modulo 90 degrees."""
    angles = np.round(angles / (np.pi / 2)) * (np.pi / 2)
    return normalize_angles(angles)


def get_parallel_rotations():
    mult90 = [0, np.pi / 2, -np.pi / 2, np.pi]
    parallel_rotations = []
    for euler in itertools.product(mult90, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations += [canonical]
    assert len(parallel_rotations) == 24
    return parallel_rotations


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.
    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1.0]))
        [1.0]
    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable
    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    E.g.:
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path
    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS4:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < _EPS4:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0