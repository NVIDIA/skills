# Dependency inventory

Do not `poetry install` into the simulator or training environment. The manifest
and Poetry 1.x lock expose imports to repository dependency scanners; runtime
versions remain owned by the installed products and are recorded per invocation.

| Direct import | Provider | License |
| --- | --- | --- |
| `yaml` / PyYAML | Verified training `.venv` | MIT |
| `torch` | Verified training `.venv`, export reset compatibility | BSD-3-Clause |
| `numpy` | Isaac Sim Python | BSD-3-Clause |
| `scipy` | Isaac Sim Python | BSD-3-Clause |
| `pxr` / OpenUSD (`usd-core` inventory) | Isaac Sim Python | Apache-2.0 with modified patent terms |
| `rclpy` | Isaac Sim ROS bridge / ROS container | Apache-2.0 |
| `geometry_msgs`, `sensor_msgs`, `std_msgs` | ROS common_interfaces | Apache-2.0 |
| `isaac_ros_deploy_interfaces` | Packaged Isaac ROS Deploy | Apache-2.0 |
| `isaacsim.*`, `omni.*`, `carb` | Isaac Sim distribution | NVIDIA product terms; see bundled licenses |

ROS dependencies are also declared in `package.xml` because they are supplied as
ROS distribution packages, not installed from PyPI. The simulator includes its
own OpenUSD build; the inventory wheel is not substituted for it.

External executables: Python (PSF), uv (MIT/Apache-2.0), Docker Engine
(Apache-2.0), NVIDIA Container Toolkit (Apache-2.0), and ffmpeg (LGPL-2.1-or-later
or GPL depending on build; recording requires an existing H.264/libx264-capable
build). Record the exact ffmpeg build. IsaacLabTraining/LEAPP, Triton, CUDA, and
Isaac Sim bring their own product dependencies and licenses; this skill does not
redistribute their binaries or assets.
