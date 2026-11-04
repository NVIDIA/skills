#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a saved policy and run bounded ROS/Isaac Sim insertion trials.

Orchestration uses the standard library. Product environments supply their own
dependencies; no source checkout is downloaded and no physical robot is opened.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from training_support import launch_environment

SCRIPTS = Path(__file__).resolve().parent
DEFAULT_IMAGE = "nvcr.io/nvidia/isaac/ros:isaac_ros_1742a70509cba44cb1943ab200bbece1-amd64"
DEFAULT_SIM_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.0"
SIM_ROOT = "/isaac-sim"
SIM_FFMPEG = "/usr/bin/ffmpeg"
SIM_ENV = """export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}"
"""
ROS_ENV = """source /opt/ros/lyrical/setup.bash
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
"""


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Session:
    def __init__(self, args):
        self.args = args
        configuration = {}
        plan_path = getattr(args, "training_plan", None)
        if plan_path:
            plan = json.loads(Path(plan_path).read_text())
            if Path(plan["training_root"]).resolve() != Path(args.training_root).resolve():
                raise ValueError("--training-plan must describe the selected --training-root")
            configuration = plan.get("environment", {})
        if getattr(args, "uv", None):
            configuration = dict(configuration, uv_executable=args.uv)
        self.configuration = launch_environment.normalize_environment(configuration)
        self.environment = launch_environment.build_environment(
            Path(args.training_root), self.configuration, eula_accepted=args.accept_eula)
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.commands = self.root / "commands.jsonl"
        self.docker = (["sudo", "-n"] if args.sudo_docker else []) + ["docker"]
        self.container = "cable-ros-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S") + f"-{os.getpid()}"
        self.sim_container = self.container.replace("cable-ros-", "cable-sim-", 1)
        self.sim_created = False
        self.children = []
        self.created = False
        self.results = []
        self.status = "running"
        self.error = None

    def record(self, command, log):
        with self.commands.open("a") as stream:
            stream.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                     "cwd": self.args.training_root, "argv": command,
                                     "log": str(log)}) + "\n")
        print(shlex.join(command), flush=True)

    def run(self, command, log, timeout=120, env=None):
        log = self.root / log
        self.record(command, log)
        with log.open("ab") as stream:
            proc = subprocess.Popen(command, cwd=self.args.training_root, stdout=stream,
                                    stderr=subprocess.STDOUT,
                                    env=self.environment if env is None else env,
                                    start_new_session=True)
            self.children.append(proc)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.stop(proc)
                raise RuntimeError(f"Command timed out; see {log}") from None
        if code:
            raise RuntimeError(f"Command exited {code}; see {log}")

    def start(self, command, log):
        log = self.root / log
        self.record(command, log)
        with log.open("ab") as stream:
            proc = subprocess.Popen(command, cwd=self.args.training_root, stdout=stream,
                                    stderr=subprocess.STDOUT, env=self.environment,
                                    start_new_session=True)
        self.children.append(proc)
        return proc

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                proc.wait(timeout=10)

    def inside(self, command):
        return self.docker + ["exec", self.container, "bash", "-c", ROS_ENV + "exec " + shlex.join(command)]

    def sim_inside(self, command):
        return self.docker + ["exec", self.sim_container, "bash", "-c",
                              SIM_ENV + "exec " + shlex.join(command)]

    def inspect_image(self, reference, log):
        self.run(self.docker + ["image", "inspect", reference], log)
        metadata = json.loads((self.root / log).read_text())
        if len(metadata) != 1 or not metadata[0]["Id"].startswith("sha256:"):
            raise ValueError(f"Could not resolve immutable image identity: {reference}")
        return metadata[0]["Id"]

    def create_simulator(self):
        a = self.args
        # The image's default UID cannot read private host session/skill files.
        # Root is confined to this owned container; only the session is writable.
        command = self.docker + ["run", "-d", "--name", self.sim_container,
            "--label", "isaac-skills=isaac-cable-insertion-policy-deployment", "--user", "0:0",
            "--gpus", "all", "--network", "host", "--shm-size", "2g",
            "-e", "ACCEPT_EULA=Y", "-e", "OMNI_KIT_ACCEPT_EULA=Yes",
            "-e", "OMNI_KIT_ALLOW_ROOT=1", "-e", "ROS_DISTRO=jazzy",
            "-e", f"ROS_DOMAIN_ID={a.domain_id}",
            "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", "-e", "FASTDDS_BUILTIN_TRANSPORTS=UDPv4",
            "-v", f"{SCRIPTS}:/skill:ro", "-v", f"{self.root}:{self.root}",
            "--entrypoint", "bash", self.sim_image_id, "-c", "sleep infinity"]
        self.sim_created = True
        self.run(command, "sim-container.log", timeout=1800)
        if a.install_sim_runtime:
            install = "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg"
            self.run(self.docker + ["exec", self.sim_container, "bash", "-c", install],
                     "sim-container.log", timeout=1800)
        self.run(self.docker + ["exec", self.sim_container, "cat", SIM_ROOT + "/VERSION"], "sim-version.txt")
        version = (self.root / "sim-version.txt").read_text().strip()
        if not version.startswith("6.0.0"):
            raise ValueError(f"Deployment requires Isaac Sim 6.0.0; image contains {version}")
        for name in ("python.sh", "kit/kit", "kit/python/bin/python3",
                     "exts/isaacsim.ros2.core/bin/isaacsim.ros2.core.check"):
            self.run(self.docker + ["exec", self.sim_container, "test", "-x", SIM_ROOT + "/" + name],
                     "sim-preflight.log")
        self.run(self.sim_inside([SIM_FFMPEG, "-version"]), "sim-preflight.log")
        self.run(self.sim_inside([SIM_FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
                  "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "16x16",
                  "-i", "/dev/zero", "-frames:v", "1", "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-f", "null", "-"]), "sim-preflight.log")
        self.run(self.docker + ["exec", self.sim_container, "dpkg-query", "-W"], "sim-packages.log")
        dump(self.root / "runtime.json", {"isaac_sim_build": version, "arguments": vars(a),
             "sim_image_id": self.sim_image_id, "runtime_image_id": self.runtime_image_id,
             "sim_container": self.sim_container, "ros_container": self.container,
             "skill_scripts": {p.name: sha256(p) for p in SCRIPTS.glob("*.py")}})

    def check_compatibility(self):
        checker = self.start(self.sim_inside([SIM_ROOT + "/isaac-sim.compatibility_check.sh",
                             "--/app/quitAfter=10", "--no-window"]), "compatibility.log")
        deadline = time.monotonic() + 180
        while True:
            output = (self.root / "compatibility.log").read_text(errors="replace")
            if "System checking result: FAILED" in output:
                raise RuntimeError("Isaac Sim compatibility check failed; see compatibility.log")
            if "System checking result: PASSED" in output:
                break
            if checker.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Isaac Sim compatibility check did not report PASSED; see compatibility.log")
            time.sleep(1)
        # A checker may remain open after its semantic result. Stop all of its
        # container processes before starting inference and the recording.
        self.run(self.docker + ["stop", "--time", "10", self.sim_container], "sim-container.log")
        self.stop(checker)
        self.run(self.docker + ["start", self.sim_container], "sim-container.log")

    def preflight(self):
        a = self.args
        if not a.accept_eula:
            raise ValueError("Explicit/remembered user EULA acceptance is required: --accept-eula")
        checkpoint = Path(a.checkpoint)
        for path in [Path(a.training_root) / ".venv/bin/python", checkpoint,
                     checkpoint.parent / "params/env.yaml", checkpoint.parent / "params/agent.yaml"]:
            if not path.is_file():
                raise ValueError(f"Required file missing: {path}")
        self.run([str(Path(a.training_root) / ".venv/bin/python"),
                  str(SCRIPTS / "training_support.py"), a.training_root], "installation.log")
        dump(self.root / "environment.json", self.configuration)
        dump(self.root / "export-runtime.json", {"arguments": vars(a),
             "skill_scripts": {p.name: sha256(p) for p in SCRIPTS.glob("*.py")},
             "training_support": {p.name: sha256(p) for p in
                 [Path(launch_environment.__file__), Path(launch_environment.__file__).with_name("runtime_checks.py")]}})
        if getattr(a, "export_only", False):
            return
        self.run(["nvidia-smi"], "preflight.log")
        self.run(self.docker + ["info"], "preflight.log")
        self.runtime_image_id = self.inspect_image(a.runtime_image, "image.json")
        self.sim_image_id = self.inspect_image(a.sim_image, "sim-image.json")
        self.create_simulator()
        self.check_compatibility()

    def export(self):
        a = self.args
        self.run([str(Path(a.training_root) / ".venv/bin/python"), str(SCRIPTS / "prepare.py"),
                  "--training-root", a.training_root, "--checkpoint", a.checkpoint,
                  "--output", str(self.root)], "prepare.log")
        command = json.loads((self.root / "export-command.json").read_text())
        env = dict(self.environment)
        env["OMNI_KIT_ACCEPT_EULA"] = "Yes"
        self.run(command, "export.log", timeout=1800, env=env)
        bundles = list((self.root / "export").rglob("DisplayPortTaskSpace.yaml"))
        if len(bundles) != 1:
            raise RuntimeError("Expected exactly one DisplayPortTaskSpace export YAML")
        self.bundle = bundles[0]
        for suffix in [".onnx", "_initial_values.safetensors"]:
            file = self.bundle.parent / ("DisplayPortTaskSpace" + suffix)
            if not file.is_file() or not file.stat().st_size:
                raise RuntimeError(f"Missing/empty export artifact {file}")
        self.run([str(Path(a.training_root) / ".venv/bin/python"), str(SCRIPTS / "validate_export.py"),
                  "--bundle", str(self.bundle)], "export-validation.log")
        dump(self.root / "export-files.json", {str(p.relative_to(self.root)): sha256(p)
                                               for p in self.bundle.parent.iterdir() if p.is_file()})

    def create_runtime(self):
        """Create the session runtime container and verify its ROS dependencies."""
        a = self.args
        command = self.docker + ["run", "-d", "--name", self.container,
            "--label", "isaac-skills=isaac-cable-insertion-policy-deployment", "--gpus", "all",
            "--network", "host", "--shm-size", "2g", "-e", f"ROS_DOMAIN_ID={a.domain_id}",
            "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", "-e", "FASTDDS_BUILTIN_TRANSPORTS=UDPv4",
            "-v", f"{SCRIPTS}:/skill:ro", "-v", f"{self.root}:/session",
            "--entrypoint", "bash", self.runtime_image_id, "-c", "sleep infinity"]
        self.created = True
        self.run(command, "container.log", timeout=1800)
        if a.install_runtime:
            install = "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ros-lyrical-isaac-ros-deploy-reference-applications ros-lyrical-rmw-fastrtps-cpp cuda-cupti-13-2 && ldconfig"
            self.run(self.docker + ["exec", self.container, "bash", "-c", install], "container.log", timeout=1800)
        self.run(self.inside(["ros2", "pkg", "prefix", "isaac_ros_deploy_reference_applications"]), "container.log")
        self.run(self.docker + ["exec", self.container, "dpkg-query", "-W"], "packages.log")

    def start_policy(self, out):
        relative = out.name
        config = "/session/" + str(self.bundle.relative_to(self.root))
        command = ["ros2", "launch", "isaac_ros_deploy_reference_applications",
            "displayport_task_space_policy.launch.py", f"config_path:={config}",
            "publish_rate:=30.0", "use_sim_time:=true", "base_frame:=world",
            "flange_pose_topic:=/displayport_insertion/sim_flange_pose",
            "socket_pose_topic:=/displayport_insertion/goal_pose", "require_socket_pose:=true",
            "tcp_offset:=0.0,0.0,0.1925", "action_blend_ratio:=1.0",
            "enable_topic:=/displayport_task_space_policy/enable", "enabled_on_start:=false",
            "target_pose_topic:=/displayport_task_space_policy/unused_tcp_target",
            f"csv_output_path:=/session/{relative}/policy-io.csv",
            "debug_policy_io_topic:=/displayport_task_space_policy/debug_io"]
        policy = self.start(self.inside(command), f"{relative}/policy.log")
        deadline = time.monotonic() + 180
        while "Triton server is live and ready" not in (out / "policy.log").read_text(errors="replace"):
            if policy.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"Policy failed readiness; see {out / 'policy.log'}")
            time.sleep(1)
        return policy

    def attempt(self, number):
        out = self.root / f"attempt-{number:02d}"
        out.mkdir()
        relative = out.name
        policy = self.start_policy(out)
        adapter = self.start(self.inside(["python3", "/skill/flange_action_adapter.py", "--output",
                                         f"/session/{relative}/flange-targets.jsonl", "--action-blend", "1.0"]),
                             f"{relative}/adapter.log")
        self.run(self.sim_inside([SIM_ROOT + "/python.sh", "/skill/sim_record.py",
                  "--output", str(out), "--asset-manifest", str(self.root / "asset-manifest.json"),
                  "--ffmpeg", SIM_FFMPEG, "--domain-id", str(self.args.domain_id), "--duration", "15"]),
                 f"{relative}/simulator.log", timeout=900)
        if policy.poll() is not None or adapter.poll() is not None:
            raise RuntimeError("ROS process exited unexpectedly during simulation")
        # Stopping the owned container terminates its processes, including docker-exec descendants.
        self.run(self.docker + ["stop", "--time", "10", self.container], "container.log")
        self.stop(policy)
        self.stop(adapter)
        result = json.loads((out / "result.json").read_text())
        self.run(self.sim_inside([SIM_FFMPEG, "-v", "error", "-i", str(out / "video.mp4"),
                  "-f", "null", "-"]), f"{relative}/video-check.log")
        result["attempt"] = relative
        self.results.append(result)
        return result

    def report(self):
        """Write session status and a report linking recorded deployment evidence."""
        dump(self.root / "status.json", {"status": self.status, "error": self.error,
                                         "attempts": self.results})
        lines = ["# Isaac cable insertion policy deployment", "", f"Status: **{self.status}**", "",
                 f"Checkpoint: `{self.args.checkpoint}`", "",
                 "Fresh LEAPP export; ROS and simulation evidence are reported separately below.", "",
                 "Export validation alone does not establish insertion success.", ""]
        if self.error:
            lines += [f"Error: {self.error}", ""]
        for row in self.results:
            error = row.get("final_position_error_m", row.get("final_error_m"))
            error_text = f"{1000 * error:.3f} mm" if error is not None else "unavailable"
            lines += [f"## {row['attempt']}", "",
                      f"Insertion: **{'passed' if row.get('success') else 'failed'}** · Final position error: **{error_text}** · Criterion: <3 mm.", "",
                      f"Policy targets: {row.get('target_count', 'unknown')} · Video frames: {row.get('frames', 'unknown')}.", "",
                      f"[Video]({row['attempt']}/video.mp4) · [Trace]({row['attempt']}/trace.json) · "
                      f"[Asset checks]({row['attempt']}/asset-verification.json) · [All metrics]({row['attempt']}/result.json)", ""]
        lines += ["## Files", "", "- `params/`: original agent/environment settings",
                  "- `export/`: newly exported LEAPP bundle", "- `asset-manifest.json`: training asset references and contract",
                  "- `commands.jsonl`: exact commands and log paths", "- `runtime.json`, `*packages.log`, `*image.json`: both container runtimes",
                  "- `compatibility.log`: simulator image compatibility result",
                  "- `export-files.json`: exported file hashes", "",
                  "Asset checks establish parity with the saved URLs resolved now; historical asset bytes were not recorded."]
        (self.root / "report.md").write_text("\n".join(lines) + "\n")

    def cleanup(self):
        for created, container, mount, log in (
                (self.sim_created, self.sim_container, str(self.root), "sim-container.log"),
                (self.created, self.container, "/session", "container.log")):
            if not created:
                continue
            try:
                self.run(self.docker + ["stop", "--time", "10", container], log)
                # Both images run a passive keepalive on restart. Fix only the
                # owned session mount, then leave the container stopped.
                self.run(self.docker + ["start", container], log)
                self.run(self.docker + ["exec", "--user", "0:0", container, "chown", "-R",
                                       f"{os.getuid()}:{os.getgid()}", mount], log)
            except Exception as exc:
                self.status = "failed during cleanup"
                self.error = (self.error or "") + f"; cleanup {container}: {exc}"
            finally:
                try:
                    self.run(self.docker + ["stop", "--time", "10", container], log)
                except Exception as exc:
                    self.status = "failed during cleanup"
                    self.error = (self.error or "") + f"; stop {container}: {exc}"
        for proc in self.children:
            self.stop(proc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", "--isaaclab", dest="training_root", required=True,
                   help="Manipulation repo's isaaclab_training export environment")
    p.add_argument("--training-plan", help="Reuse the training plan's selected tools and environment")
    p.add_argument("--uv", help="Explicit absolute uv executable; overrides the training plan")
    p.add_argument("--export-only", action="store_true", help="Validate and export without ROS or a deployment simulator")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sim-image", default=DEFAULT_SIM_IMAGE, help="Isaac Sim 6.0.0 Docker image or digest")
    p.add_argument("--install-sim-runtime", action="store_true", help="Install ffmpeg inside the new simulator container")
    p.add_argument("--output", required=True, help="New directory; never reuse an earlier run")
    p.add_argument("--runtime-image", default=DEFAULT_IMAGE)
    p.add_argument("--install-runtime", action="store_true", help="Install binary packages inside the new container")
    p.add_argument("--sudo-docker", action="store_true")
    p.add_argument("--accept-eula", action="store_true")
    p.add_argument("--domain-id", type=int, default=211)
    p.add_argument("--attempts", type=int, choices=range(1, 4), default=1)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    for name in ["training_root", "checkpoint", "output", "training_plan", "uv"]:
        if getattr(a, name):
            setattr(a, name, str(Path(getattr(a, name)).expanduser().resolve()))
    if not 0 <= a.domain_id <= 232:
        p.error("Use a dedicated ROS domain ID from 0 through 232")
    session = Session(a)
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Deployment interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        session.preflight()
        if a.preflight_only:
            session.status = "preflight passed; export and simulation not run"
        elif a.export_only:
            session.export()
            session.status = "export validated; ROS and deployment simulation not run"
        else:
            session.export()
            session.create_runtime()
            for number in range(1, a.attempts + 1):
                if number > 1:
                    session.run(session.docker + ["start", session.container], "container.log")
                result = session.attempt(number)
                if result.get("success"):
                    session.status = "success"
                    break
            else:
                session.status = "completed without insertion success"
    except (Exception, KeyboardInterrupt) as exc:
        session.status = "failed"
        session.error = str(exc) or type(exc).__name__
        print(session.error, file=sys.stderr)
    finally:
        session.cleanup()
        session.report()
    print(f"Report: {session.root / 'report.md'}", flush=True)
    return 0 if session.status.startswith(("success", "preflight passed", "export validated")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
