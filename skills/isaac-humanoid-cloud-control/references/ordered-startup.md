# Ordered startup and mission workflow

Treat startup timing as a hard phase gate. Complete cloud startup and prepare
the sourced Isaac ROS launch shell, including all arguments,
before pressing **Play** in Isaac Sim. The required order is:

`cloud healthy -> browser viewer ready -> headless Isaac Sim ready and paused -> ROS launch staged -> Play in streamed UI -> immediately launch the Isaac ROS stack`

Do not insert diagnostics, configuration work, or operator wait time between
**Play** and the ROS launch. Leaving the scene playing without the G1 control
stack can make the humanoid unstable and no longer controllable. If the launch
is delayed and instability appears, do not release the gantry or attempt a
mission; stop the robot launch, stop the timeline, reload/reset the sample
scene, and repeat the ordered startup.

1. Run the host preflight and capture its discovered ISAAC_PATH, ISAAC_ROS_WS,
   and WEB_VIEWER_CONTEXT before asking for any host paths. Then collect the
   cloud-stack host, robot name, and simulated spawn pose. Resolve the host-side
   map PNG/YAML from `$SKILL_DIR/assets`. Prepare and record the dedicated
   runtime `MC_WORKSPACE` through the cloud workspace procedure linked from
   `SKILL.md`; defaults and Compose come from the owning skills. Use an approved
   destination when the operator already supplied one; otherwise defer the
   destination until the live scene suggestions in step 12 are ready. Treat a
   selected suggestion as the operator-supplied destination, and never infer
   authorization to move real hardware.
2. Inspect existing processes, containers, ports, Isaac Sim installation,
   streaming ports, GPU encode support, and workspace state read-only. Do not
   stop an existing simulator, viewer, or cloud stack without permission.
   Require loopback port 8226 to be unused before launching the demo-owned
   Isaac Sim Python server; do not reuse an unknown listener.
3. Run the host preflight:

   ~~~bash
   python3 <skill-dir>/scripts/preflight.py \
     --isaac-path "$ISAAC_PATH" \
     --isaac-ros-ws "$ISAAC_ROS_WS" \
     --map /absolute/path/to/occupancy_map.yaml \
     --web-viewer-context /absolute/path/to/IsaacSim/tools/docker/web-viewer \
     --cloud-host <cloud-stack-host> --check-cloud
   ~~~

   Omit --check-cloud until the cloud stack should be listening. If the CLI or
   Docker setup is missing, apply the CLI installation and Docker setup reference
   linked from `SKILL.md`, finish the bootstrap, and rerun this check before launch.
4. Ensure the release package and Isaac ROS Deploy extension are available.
   Follow the binary or source setup in the release reference only after the
   user selects the delivery path and approves host changes.
5. Follow the cloud workspace procedure to copy the map assets and preview/apply
   map and fleet configuration through the owning adapters. Require the same
   map and robot identity as the G1 launch. Do not hand-edit defaults or Compose.
6. Start cloud services through bring-up-cloud-stack without --with-sim, using
   the prepared workspace and recorded isolated Compose project:

   ~~~bash
   python3 "$SKILL_DIR/references/bring-up-cloud-stack/scripts/run.py" --dry-run
   python3 "$SKILL_DIR/references/bring-up-cloud-stack/scripts/run.py"
   ~~~

   Preserve the exported `MC_WORKSPACE` and `COMPOSE_PROJECT_NAME` in both calls.
   Follow the owning skill's execution gates. Verify MQTT, Mission Dispatch,
   Mission Control, database, WPG, and cuOpt.
7. Require NVIDIA's official Isaac Sim 6.0 `tools/docker/web-viewer` source.
   A standalone 6.0.1 runtime archive does not include it. If installation of
   mission apps was authorized and no matching checkout is discoverable, use a
   shallow checkout of the official `isaac-sim/IsaacSim` repository at tag
   `v6.0.1` with Git LFS smudging disabled, then verify the resolved tag/commit
   and Dockerfile before building. Do not use `main` or an unpinned clone.
   Start it with the bundled viewer-only
   `assets/browser-viewer.compose.yaml` so it connects to the standalone
   streaming app without launching a second Isaac Sim. Obtain the URL from its
   logs; default to `http://<sim-host>:8210`. Verify the URL returns the viewer
   page, then immediately provide it to the user as a clickable live-stream
   link in a commentary update; do not make the user infer it from logs or wait
   until the final report. Use `http://127.0.0.1:8210` only when the user is on
   the simulator host. In a remote SSH session, derive the viewer host from the
   server-address field (the third field) of `SSH_CONNECTION`; do not select an
   arbitrary address from `hostname -I` or the viewer logs. Pass an explicit
   `--viewer-host` only when the operator supplies a different routed hostname
   or IP. Export the discovered address as `WEB_VIEWER_HOST` before building the
   bundled viewer Compose service; the official viewer compiles its signaling
   and media host into the JavaScript at image-build time. A page built with
   `127.0.0.1` can return HTTP 200 remotely but remain at `WAITING FOR STREAM`.
   When Codex browser panels are available, open the same remote URL with
   `open_in_codex`. Otherwise open it in a Chromium-based browser. Do not expose
   the viewer or WebRTC ports directly to the public
   Internet. If source checkout was not authorized or the exact matching tag
   cannot be verified, stop and report the missing dependency instead of using
   an unpinned clone, reverting to a local GUI, or substituting Foxglove for the
   rendered Isaac Sim view. Treat `npm audit` findings from the upstream viewer
   build as reportable evidence; do not silently rewrite its lockfile during a
   demo build.
8. Use the tutorial's production prebuilt USD exactly; it is a remote Isaac
   Sim asset, not a required local file:

   ~~~bash
   export DEPLOY_USD=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/ROS2/Scenario/g1_ros_navigation.usd
   export PYTHONPATH="$ISAAC_ROS_WS/src/isaac_ros_deploy/isaac_deploy/isaac_ros_deploy_isaac_sim_extension:$PYTHONPATH"
   export LD_LIBRARY_PATH="$ISAAC_PATH/exts/isaacsim.ros2.core/jazzy/lib:$LD_LIBRARY_PATH"
   export ROS_DISTRO=jazzy
   export ROS_DOMAIN_ID=<clean-demo-domain>
   export ROS_LOCALHOST_ONLY=1
   export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   test -x "$ISAAC_PATH/exts/isaacsim.ros2.core/bin/isaacsim.ros2.core.check"
   "$ISAAC_PATH/isaac-sim.streaming.sh" --no-window \
     --ext-folder "$ISAAC_ROS_WS/src/isaac_ros_deploy/isaac_deploy/isaac_ros_deploy_isaac_sim_extension" \
     --ext-folder "<skill-dir>/extensions" \
     --enable isaac_ros_deploy_isaac_sim_extension \
     --enable isaacsim.ros2.sim_control \
     --enable isaacsim.code_editor.python_server \
     --enable humanoid_cloud_control.autoplay \
     --/exts/isaacsim.code_editor.python_server/host=127.0.0.1 \
     --/exts/isaacsim.code_editor.python_server/port=8226 \
     --/exts/omni.kit.livestream.app/primaryStream/enableEventTracing=false \
     --/exts/isaac_ros_deploy_isaac_sim_extension/usd="$DEPLOY_USD"
   ~~~

   `isaac-sim.sh` with the same extension flags is the upstream tutorial's
   interactive form; retain `isaac-sim.streaming.sh --no-window` only for this
   skill's required browser-streamed deployment. Keep rendering enabled. Leave
   WebRTC event tracing disabled for normal demos so `NvStreamer-*.etli`
   diagnostics do not accumulate in the launch working directory. Temporarily
   set `enableEventTracing=true` only when investigating stream encoding,
   latency, packet loss, or QoS, and keep the resulting traces out of the
   repository.
   Require `app ready`, a connected browser viewport, the loopback Python
   server at 127.0.0.1:8226, and the timeline paused. Never expose 8226 beyond
   the simulator host.
   In parallel, prepare a separate
   long-lived isaac-ros activate shell: source the workspace and stage
   unitree_g1_cloud_control.launch.py with hardware_type:=isaacsim, the
   matching map, cloud MQTT host, robot name, and measured
   spawn pose. Use `enable_camera_localization:=false` and `use_static_tf:=true`;
   set `controller_group:=agile_velocity` so the simulation-only navigation
   path does not require GR00T policy assets. Do not add Visual SLAM or
   camera-localization components. Do not press
   **Play** until this command is ready to execute.

   Configure the G1 MQTT host, port, transport, and authentication to match the
   resolved upstream broker configuration. Supply the same nonempty username
   and password when authentication is enabled, using the approved credential
   mechanism; do not print secrets in progress reports or save them in the
   skill. Omit credential arguments only if the selected broker is explicitly
   configured for anonymous access. Do not pass empty assignments such as
   `mqtt_username:=`, which ROS 2 rejects. Verify a successful MQTT connection
   before any balance or gantry action. For code 5 (`not authorized`), check
   credential agreement and broker identity; do not infer that authentication
   itself is stale or disable it as recovery.
   Follow the tutorial order: start Isaac Sim, wait for `app ready`, press
   Play, then run the G1 launch immediately. The sample USD already supplies
   the Deploy actuator and ROS 2 bridge graphs; do not bake a replacement USD.
   Rerun the host preflight after both streaming components start:

   ~~~bash
   python3 <skill-dir>/scripts/preflight.py \
     --isaac-path "$ISAAC_PATH" --isaac-ros-ws "$ISAAC_ROS_WS" \
     --map /absolute/path/to/occupancy_map.yaml \
     --web-viewer-context /absolute/path/to/IsaacSim/tools/docker/web-viewer \
     --sim-host <sim-host> --check-streaming --check-scene-server
   ~~~
9. For a headless end-to-end run, enable the bundled
   `humanoid_cloud_control.autoplay` extension. It starts only the Isaac Sim
   timeline after startup and extends the short sample animation range to 24
   hours without looping, which keeps ROS simulation time monotonic. Before
   Play, it also applies `ROS_DOMAIN_ID` to authored ROS 2 Context nodes in the
   in-memory stage so the simulator and activated shell share the selected
   domain. It does not release the virtual gantry, set the policy gate, publish
   velocity, or submit a mission. Start the staged G1
   launch before the extension's startup delay elapses. Keep
   `isaacsim.ros2.sim_control` enabled for observable, manual timeline control.
   If autoplay is deliberately omitted, start the paused timeline through the
   built-in ROS 2 simulation-control service, then immediately execute the
   staged ROS launch command:

   ~~~bash
   ros2 service call /set_simulation_state \
     simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"
   ~~~

   Require a successful result, then confirm /clock, joint state, and IMU data
   promptly. Use the browser helper only as a fallback when this service is not
   supplied by the selected Isaac Sim build. Neither path releases the virtual
   gantry as a side effect.
10. Run the guarded helper for the simulation-only `agile_velocity` group.
   It verifies live simulation identity before enabling AGILE or publishing any
   velocity. Never run manual controller-enable or velocity commands ahead of
   this interlock. A pose held by the virtual gantry is not evidence of balance.

   The helper must establish safety-controller readback `1.0`, fresh joint and
   IMU data, active policy output, and visibly bounded attitude. Before invoking
   the helper, state the simulation target and request explicit approval for
   this release attempt. A generic end-to-end demo request is insufficient.
   If this specific attempt is already explicitly approved, use that approval;
   otherwise keep the gantry engaged while waiting for the operator's answer.
   The same host must run Isaac Sim and the activated ROS environment. Set
   `ROS_LOCALHOST_ONLY=1`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, and an
   explicit nonzero dedicated `ROS_DOMAIN_ID` before starting either process.
   Use `rmw_fastrtps_cpp` without custom discovery profiles or static peers.
   Resolve the pinned scene adapter in the activated environment as well.
   The helper requires the live sample USD and playing timeline from the
   loopback scene server, a matching simulator ROS domain, simulation time,
   only the Isaac Sim controller hardware plugin, and exactly one gantry service
   provider named `isaacsim_launcher`. Missing or ambiguous evidence blocks all
   controller mutations; do not bypass the gate with manual ROS commands.
   These checks prevent accidental target mixups on a trusted local host; DDS
   node names are not authentication against a hostile local process.
   Run it under the same
   activated container user and middleware environment as the G1 launch. Do not
   run it as container `root` when the activated shell is a non-root user; Fast
   DDS shared-memory permissions can split discovery despite a matching domain:

   ~~~bash
   python3 <skill-dir>/scripts/guarded_gantry_release.py \
     --allow-simulation-gantry-release
   ~~~

   The helper must set and read back `blend_ratio=1.0`, publish zero velocity,
   require fresh joint/IMU samples and continuously bounded tilt/angular rate,
   disable the gantry through NVIDIA's supported
   `/launcher/gantry/set_enabled` service, and require a second unsupported
   stability window. On any post-release limit violation it must re-enable the
   gantry and fail. If the log confirms successful automatic re-engagement, no
   mission is active, and the timeline, controller, services, and sensors remain
   healthy, report the failed attempt and ask for explicit approval for one
   simulation-only retry. Keep the gantry engaged until approval arrives, then
   recheck readiness and retain the same tilt/angular-rate limits:

   ~~~bash
   python3 <skill-dir>/scripts/guarded_gantry_release.py \
     --allow-simulation-gantry-release \
     --stabilization-seconds 10.0 \
     --post-release-seconds 10.0
   ~~~

   This retry is the full bounded allowance; do not attempt a third release. If
   re-engagement is unconfirmed or the retry fails, reset the scene through the
   startup timing gate. Do not submit until
   the helper exits successfully. Do not inventory or present destinations until
   the helper reports that unsupported balance held for its full post-release
   window. If the user requested manual **G** release instead, require equivalent
   fresh sensor and unsupported-stability evidence. Do not reuse this helper for
   real hardware.
11. Run the runtime readiness helper inside the activated environment:

    ~~~bash
    python3 <skill-dir>/scripts/check_runtime.py --base-frame pelvis
    ~~~

    Also require the configured robot to be online, IDLE, error-free, and
    position-initialized in Mission Control. Do not submit during another
    active mission.
12. After runtime and cloud readiness pass, inventory the actual live USD stage
    through isaac-sim-remote and generate object-aware navigation suggestions:

    ~~~bash
    python3 "$SKILL_DIR/references/isaac-sim-remote/scripts/run.py" \
      --host 127.0.0.1 --port 8226 \
      --file "$SKILL_DIR/scripts/scene_inventory.py"
    # Use the unique scene_inventory_path reported above on the simulator host.
    python3 <skill-dir>/scripts/suggest_navigation_missions.py \
      --inventory /absolute/reported/scene_inventory_path \
      --map /absolute/path/to/occupancy_map.yaml \
      --robot-x <measured-x> --robot-y <measured-y> --max-suggestions 5
    ~~~

    The inventory must report prim paths, semantic labels when authored, asset
    metadata, world-space centers, and bounds. Treat semantic labels as the
    strongest classification, then prim names, then asset metadata. Clearly
    label name/asset classifications as heuristics. Never infer shelves or
    other object classes from occupancy geometry when live inventory is
    available. The suggestion helper checks map occupancy and footprint
    clearance but not Nav2 reachability. Run the active Nav2 planner for every
    suggestion shown; discard unreachable poses. Present 3-5 diverse, concise
    navigation-only options with object name, exact pose, classification source,
    and the clickable live-view URL. Ask the operator to select or amend one.
    If no recognizable objects are present, say so and offer coordinate-based
    navigation; do not invent semantic landmarks or suggest manipulation.

13. After the operator approves a destination, check the endpoint against the occupancy grid
    with robot-footprint clearance and verify the route from the measured pose
    is reachable with the active Nav2 planner. Run the submission helper in
    preview mode first. Present the robot ID, mission ID, exact route, and
    timeout to the operator and obtain explicit approval to submit that payload.
    Selecting a destination alone is not submission approval. Keep the same
    explicit mission ID for preview and submission. If the route, target, or
    readiness changes while waiting, revalidate and present a new preview for
    approval. Only then invoke the helper with --submit. If automated validation cannot
    establish clearance and reachability, do not submit and do not silently
    substitute another destination.

    ~~~bash
    python3 <skill-dir>/scripts/submit_navigation_mission.py \
      --host <cloud-stack-host> --robot g1_0 --mission-id <approved-mission-id> \
      --x <approved-x> --y <approved-y> --theta <approved-yaw> --map-id map \
      --validated-free-space --submit
    ~~~

    The helper targets the release-4.6 Mission Control API on port 8050 and
    verifies robot readiness through the Mission Database controller on port
    5003. Mission Control may take tens of seconds to solve a route. If the
    HTTP response times out, resolve the mission ID through Mission Database;
    never blindly retry an ambiguous POST. Inspect the stored mission tree
    because WPG/cuOpt can snap requested route points to the configured graph.

14. Keep the browser WebRTC view open and visible throughout balance, gantry
    release, navigation, and completion. Capture evidence from the browser
    viewport, not from a nonexistent local Isaac Sim window. Keep the clickable
    live-stream URL visible in user-facing updates whenever reporting demo
    progress. If the stream disconnects or the URL becomes unreachable, say so
    immediately and restore it before claiming the user can watch live. If the
    page remains at `WAITING FOR STREAM`, inspect the built viewer JavaScript and
    require its `signalingServer` and `mediaServer` to equal `WEB_VIEWER_HOST`;
    then require a remote TCP connection on 49100 and the UDP 47998 listener.
    Rebuild only the demo-owned viewer when the compiled host is stale. For
    telemetry,
    open Foxglove Studio or its browser client and connect to
    ws://<robot-stack-host>:8765. Import
    foxglove_layouts/foxglove_layout_cloud_control.json from the installed
    bringup package when available.
15. Read `skills.submit-and-monitor-mission.skill_md` from the dependency
    manifest and apply its verification sections to the mission ID returned in
    step 13. Do not call its submission helper or submit a second mission.
    Report success only when the mission
    reaches COMPLETED, Dispatch agrees, the robot returns online and IDLE
    without errors, pose samples show motion, and the live view agrees.
    Inspect the stored mission tree because WPG/cuOpt may expand one requested
    pose into several graph waypoints and snap the terminal pose. During the
    final approach, do not treat circling or a quiet `/cmd_vel` sample alone as
    failure. Query `/navigate_through_poses/_action/status`, the configured goal
    checker, Mission Database, and robot pose together. Status `2` means the
    Nav2 action is still executing; status `4` is succeeded. Never inject a
    manual zero command while the action remains executing merely because the
    learned input pipeline reports a stale command. In the validated 6.0.1/
    release-4.6 run, the solver expanded the forklift target into four
    waypoints and final convergence took about 6 minutes 31 seconds before all
    authoritative states completed.

