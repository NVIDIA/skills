# Demo teardown

Use this procedure only after the operator explicitly asks to clean up the
stack, tear down the demo, or otherwise clearly requests teardown.

1. Query Mission Database for the active robot and mission. If a mission is
   active, cancel or stop it before shutting down the robot runtime. Stop any
   manual velocity publisher.
2. Stop the G1 robot launch and headless Isaac Sim processes started for this
   demo.
3. Resolve the exact Compose project and configuration labels before removal.
   Tear down only the demo-owned browser-viewer and cloud projects. Remove their
   stopped containers and project networks rather than leaving them stopped.
4. Verify the streaming, WebRTC, scene-server, MQTT, Mission Control, and demo
   ROS ports and processes are no longer listening or running.
5. Leave `isaac_ros_dev_container` running for reuse. Preserve the recorded
   runtime workspace and dependency checkouts; removing containers does not
   authorize deleting maps, configuration, or evidence.

Do not stop a pre-existing shared stack. Do not delete shared images, maps,
models, workspaces, persistent database volumes, or credentials unless the
operator separately requests data deletion. Report the exact demo resources
removed, the reusable resources preserved, and that the previously supplied
live-stream URL is no longer available.
