# Unitree G1 cloud control — public release 4.6

Source: [NVIDIA Isaac ROS cloud-control tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_cloud_control_bringup/index.html), checked 2026-08-20.

## Environment

The tutorial's recommended Docker configuration includes the `realsense` and
`gr00t_workflow` additional image keys. In the Isaac Sim path, retain those
documented image layers but keep `enable_camera_localization:=false` and do not
submit cloud manipulation tasks: Isaac Sim supports cloud navigation only.

Prefer the public binary path for the G1 package:

```bash
isaac-ros activate
sudo apt-get update
sudo apt-get install -y ros-jazzy-isaac-ros-unitree-g1-cloud-control-bringup
```

Isaac ROS Deploy's Isaac Sim extension is source-only. Clone and build only
`isaac_ros_deploy` release-4.6 when it is absent; do not clone G1 bringup
sources just to run the binary workflow:

```bash
cd "${ISAAC_ROS_WS}/src" && \
  git clone -b release-4.6 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy.git isaac_ros_deploy
rosdep update && rosdep install --from-paths "${ISAAC_ROS_WS}/src/isaac_ros_deploy" --ignore-src -y
cd "${ISAAC_ROS_WS}" && \
  colcon build --symlink-install --packages-up-to isaac_ros_deploy_isaac_sim_extension
source install/setup.bash
```

In every activated terminal that uses a Debian bringup package, source:

```bash
source /opt/ros/jazzy/setup.bash
source "${ISAAC_ROS_WS}/install/setup.bash"  # only after building Deploy
```

## Isaac Sim navigation

Isaac Sim supports cloud-controlled navigation only, not GR00T manipulation.
Use the tutorial's prebuilt production scene rather than requiring a local USD:

```bash
export DEPLOY_USD=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/ROS2/Scenario/g1_ros_navigation.usd
export PYTHONPATH="${ISAAC_ROS_WS}/src/isaac_ros_deploy/isaac_deploy/isaac_ros_deploy_isaac_sim_extension:${PYTHONPATH}"
export LD_LIBRARY_PATH="${ISAAC_PATH}/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH}"
export ROS_DISTRO=jazzy
export ROS_DOMAIN_ID=<clean-demo-domain>
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
"${ISAAC_PATH}/isaac-sim.sh" \
  --ext-folder "${ISAAC_ROS_WS}/src/isaac_ros_deploy/isaac_deploy/isaac_ros_deploy_isaac_sim_extension" \
  --ext-folder "<skill-dir>/extensions" \
  --enable isaac_ros_deploy_isaac_sim_extension \
  --enable isaacsim.ros2.sim_control \
  --enable humanoid_cloud_control.autoplay \
  --/exts/isaac_ros_deploy_isaac_sim_extension/usd="${DEPLOY_USD}"
```

The USD already includes the Deploy actuator and ROS 2 bridge graphs. Wait for
`app ready`, press Play, then launch `unitree_g1_cloud_control.launch.py` with
`hardware_type:=isaacsim`, `use_sim_time:=True`, camera localization disabled,
static TF enabled, the sample map, cloud MQTT host, and the measured spawn pose.
For a navigation-only Isaac Sim demo without the optional GR00T policy assets,
pass `controller_group:=agile_velocity`; its policy gate is
`/safety_controller` (not `/safety_controller_lower_body`).

The cloud stack uses the pinned owning skill's broker configuration. Match the
G1 MQTT host, port, transport, and credentials to that configuration; preserve
upstream authentication. Omit credential arguments only for an explicitly
anonymous broker. Never pass empty `mqtt_username:=` or `mqtt_password:=`
assignments, which ROS 2 rejects. Keep credentials out of skill files and
reports, and restrict broker ports to the host or a trusted private network.

Use the same dedicated `ROS_DOMAIN_ID` in Isaac Sim and all activated ROS
shells. Before startup, verify that the selected domain has no existing
`/clock` or transient-local `/robot_description` publisher; a stale description
can make `ros2_control_node` initialize from a URDF without its `ros2_control`
tag.

For a headless end-to-end run, the skill supplies the narrowly scoped
`humanoid_cloud_control.autoplay` extension. It starts only the simulation
timeline after Isaac Sim loads and extends the short sample animation range to
24 hours without looping so ROS time stays monotonic. It also applies the
environment-selected domain to ROS 2 Context nodes in the in-memory stage
before Play. Keep
`isaacsim.ros2.sim_control` enabled for manual timeline control. If autoplay is
intentionally omitted, start the timeline after staging the G1 launch with:

```bash
ros2 service call /set_simulation_state \
  simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"
```

## Policy and gantry sequence

In sourced Isaac ROS terminals:

```bash
ros2 param set /safety_controller blend_ratio 1.0
ros2 param get /safety_controller blend_ratio
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'
```

Verify active balancing from live pose and IMU before proceeding. The tutorial
uses a focused Isaac Sim viewport and the `G` key to disengage the virtual
gantry. Stop the manual velocity publisher before sending a navigation mission.

The installed Isaac ROS Deploy extension also exposes the supported
`/launcher/gantry/set_enabled` `std_srvs/SetBool` service that drives the same
`VirtualGantry.enable()` / `disable()` implementation as the hotkey. Obtain explicit
operator approval for each virtual gantry release attempt before invoking the
skill's guarded helper; a broad demo request is insufficient. Invoke the service
only through that helper after AGILE readback, fresh sensor checks, a bounded
pre-release stabilization window, and zero velocity. Require a bounded
post-release window, automatically re-enable the gantry on failure, and present
navigation destinations only after the helper succeeds. Bring-up-only,
inspection-only, and real-hardware requests do not grant this authorization.
Never apply this automation to real hardware. Run the helper as the same
activated container user as the G1 launch; mixing root and non-root ROS processes
can break Fast DDS shared-memory discovery even when `ROS_DOMAIN_ID` matches.
After a confirmed automatic re-engagement, allow at most one simulation-only
retry after reporting the failure and obtaining separate operator approval.
Retain the same tilt/angular-rate limits and a 10-second pre-release
stability window. Require a healthy timeline, controller, fresh sensors, and no
active mission first. Reset the scene instead if re-engagement was not confirmed
or the retry fails.

Mission submission uses Mission Control's release-4.6
`/api/v1/mission/submit_mission` endpoint on port 8050. Canonical robot and
mission lifecycle records are exposed by the Mission Database controller on
port 5003. Route solving can outlast a short HTTP client timeout; query the
requested mission ID before considering a retry. The stored mission tree is
the authoritative solved route and can contain WPG/cuOpt graph-snapped points.
