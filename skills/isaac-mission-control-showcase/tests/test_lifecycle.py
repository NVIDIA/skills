# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline regressions: python3 -m unittest discover -s <skill>/tests -v."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR / "scripts"))
import run_state
import showcase
import doctor


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = self.temp.name
        self.manifest = run_state.initialize(
            self.work, robot="carter01", mission_id="demo", compose_project="test",
            carter_container="test-carter", ports={"database_controller_port": 5003},
        )
        self.robot = {"name": "carter01", "lifecycle": "ALIVE", "status": {
            "online": True, "position_initialized": True, "state": "IDLE", "errors": {},
            "pose": {"x": 1.0, "y": 0.0, "theta": 0.0},
        }}
        self.mission = {"name": "demo", "status": {"state": "COMPLETED"}}
        self.health = {"isaac_running": True, "processes_healthy": True, "oom": False}

    def finalize(self, code=0):
        args = argparse.Namespace(work_dir=self.work, exit_code=code, demo=True,
                                  phase="mission-running", failed_command="test failure")
        with patch.object(showcase, "runtime_health", return_value=self.health), \
                patch.object(showcase, "fetch_json", side_effect=[self.mission, self.robot]):
            return showcase.finalize(args)

    def test_acceptance_requires_each_observation(self):
        for field, values in {
            "robot_healthy": [False, None], "oom": [True, None],
            "processes_healthy": [False, None], "isaac_running": [False, None],
            "mission_state": ["FAILED", None], "robot_state": ["ON_TASK", None],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as work:
                    run_state.initialize(work)
                    run_state.record_motion(work, {"motion_confirmed": True})
                    args = dict(mission_state="COMPLETED", robot_state="IDLE",
                                robot_healthy=True, oom=False, processes_healthy=True,
                                isaac_running=True)
                    args[field] = value
                    self.assertFalse(run_state.accept(work, **args)["accepted"])

    def test_finalization_checks_robot_and_mission_identity(self):
        for index, (target, key, value) in enumerate([
            (self.robot, "name", "wrong"), (self.robot, "lifecycle", "DELETED"),
            (self.robot["status"], "online", False),
            (self.robot["status"], "position_initialized", False),
            (self.robot["status"], "errors", None),
            (self.mission, "name", "another-mission"),
        ]):
            with self.subTest(key=key, value=value):
                self.work = str(Path(self.temp.name) / str(index))
                run_state.initialize(self.work, **{k: v for k, v in self.manifest.items()
                                                  if k not in ["state", "work_dir"]})
                run_state.record_motion(self.work, {"motion_confirmed": True})
                previous = target[key]
                target[key] = value
                self.assertEqual(self.finalize(), 1)
                self.assertFalse(json.loads(run_state.result_path(self.work).read_text())[
                    "acceptance"]["accepted"])
                target[key] = previous

    def test_simulator_exit_during_final_database_queries_cannot_pass(self):
        process = subprocess.Popen(["sleep", "30"])
        self.addCleanup(process.wait)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        showcase.record_isaac(argparse.Namespace(work_dir=self.work, pid=process.pid))
        run_state.record_motion(self.work, {"motion_confirmed": True})
        args = argparse.Namespace(work_dir=self.work, exit_code=0, demo=True,
                                  phase="mission-running", failed_command="")
        def fetch(url):
            if "/mission/" in url:
                return self.mission
            process.terminate()
            process.wait(timeout=5)
            return self.robot
        with patch.object(showcase, "fetch_json", side_effect=fetch), \
                patch.object(showcase.subprocess, "check_output", side_effect=[
                    '{"Running": true, "OOMKilled": false}',
                    '{"processes": {"nav": true}, "oom_events": {"oom": 0, "oom_kill": 0}}',
                ]):
            self.assertEqual(showcase.finalize(args), 1)
        artifact = json.loads(run_state.result_path(self.work).read_text())
        self.assertFalse(artifact["acceptance"]["accepted"])
        self.assertFalse(artifact["health"]["isaac_running"])

    def test_missing_final_database_observation_fails(self):
        args = argparse.Namespace(work_dir=self.work, exit_code=0, demo=True,
                                  phase="mission-running", failed_command="")
        with patch.object(showcase, "runtime_health", return_value=self.health), \
                patch.object(showcase, "fetch_json", side_effect=OSError("unavailable")):
            self.assertEqual(showcase.finalize(args), 1)

    def test_motion_persists_before_mission_failure_and_after_stationary_sample(self):
        robot = copy.deepcopy(self.robot)
        count = 0
        def fetch(url):
            nonlocal count
            if "/mission/" in url:
                return {"name": "demo", "status": {
                    "state": "FAILED" if count == 2 else "RUNNING"}}
            count += 1
            robot["status"]["pose"]["x"] = 0.0 if count == 1 else 1.0
            return copy.deepcopy(robot)
        args = argparse.Namespace(work_dir=self.work, timeout=30,
                                  database_controller_port=5003, robot_name="carter01", mission_id="demo")
        with patch.object(showcase, "fetch_json", side_effect=fetch), \
                patch.object(showcase.time, "sleep"):
            self.assertEqual(showcase.wait_demo_complete(args), 1)
        self.assertTrue(run_state.load(self.work)["motion"]["motion_confirmed"])
        run_state.record_motion(self.work, {"motion_confirmed": False})
        self.assertEqual(self.finalize(1), 1)
        self.assertTrue(json.loads(run_state.result_path(self.work).read_text())["motion"]["motion_confirmed"])

    def test_health_rejects_child_oom_and_missing_checks(self):
        for state, observed in [
            ({"Running": True, "OOMKilled": False}, {"processes": {"nav": True}, "oom_events": {"oom": 1, "oom_kill": 1}}),
            ({"Running": True, "OOMKilled": False}, {"processes": {"nav": False}, "oom_events": {"oom": 0, "oom_kill": 0}}),
            ({"Running": True}, {}),
            ({"Running": False, "OOMKilled": True}, {}),
        ]:
            with self.subTest(state=state, observed=observed), \
                    patch.object(showcase, "process_running", return_value=True), \
                    patch.object(showcase.subprocess, "check_output", side_effect=[json.dumps(state), json.dumps(observed)]):
                health = showcase.runtime_health(self.manifest)
                self.assertFalse(health["processes_healthy"] is True and health["oom"] is False)

    def test_health_does_not_reuse_liveness_from_before_docker_checks(self):
        running = True
        def docker(argv, **kwargs):
            nonlocal running
            if argv[1] == "inspect":
                return '{"Running": true, "OOMKilled": false}'
            running = False
            return '{"processes": {"nav": true}, "oom_events": {"oom": 0, "oom_kill": 0}}'
        with patch.object(showcase, "process_running", side_effect=lambda _: running), \
                patch.object(showcase.subprocess, "check_output", side_effect=docker):
            self.assertFalse(showcase.runtime_health(self.manifest)["isaac_running"])

    def test_missing_cgroup_counters_fail(self):
        with patch.object(showcase, "process_running", return_value=True), \
                patch.object(showcase.subprocess, "check_output", side_effect=[
                    '{"Running": true, "OOMKilled": false}',
                    subprocess.CalledProcessError(1, "docker exec"),
                ]):
            self.assertIsNone(showcase.runtime_health(self.manifest)["oom"])

    def test_process_identity_rejects_reused_pid(self):
        args = argparse.Namespace(work_dir=self.work, pid=os.getpid())
        showcase.record_isaac(args)
        isaac = run_state.load(self.work)["isaac"]
        self.assertTrue(showcase.process_running(isaac))
        isaac["start_ticks"] = "wrong"
        self.assertFalse(showcase.process_running(isaac))


# All hardware/service boundaries are stubbed. Lifecycle, traps, motion,
# finalization, and cleanup execute the package's real implementations.
BOUNDARIES = r'''
resolve_runtime_entrypoints() { BRING_UP_CLOUD="$TEST_BIN/cloud"; }
load_cloud_defaults() {
  export MQTT_PORT_TCP=1883 MQTT_PORT_WEBSOCKET=9001 POSTGRES_DATABASE_PORT=5432
  export DATABASE_API_PORT=5002 DATABASE_CONTROLLER_PORT=5003 CUOPT_PORT=5050 MC_PORT=8050
  export MQTT_TRANSPORT=websockets MQTT_USERNAME=fixture MQTT_PASSWORD=fixture MQTT_REQUIRED_PORT=9001
}
run_doctor() { if [[ "$TEST_SCENARIO" == preflight-failure ]]; then return 7; fi; }
check_no_running_containers() { :; }
prefetch_runtime_images() { :; }
prepare_runtime() { :; }
restore_isaac() { :; }
wait_cloud_ready() { :; }
verify_active_map_consumers() { :; }
start_static_tf() { :; }
ensure_scan_topic() { :; }
wait_nav2_ready() { :; }
probe_runtime_consistency() { :; }
wait_mc_robot() { :; }
'''

FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
p = Path(os.environ['TEST_BIN']) / 'containers.json'
containers = json.loads(p.read_text()) if p.exists() else {}
if args[0] == 'inspect':
 name = args[-1]
 if name not in containers:
  print('Error: No such object: '+name,file=sys.stderr); sys.exit(1)
 item = containers[name]
 print(json.dumps(item['State'] if '--format' in args else [item]))
elif args[0] == 'run':
 name = args[args.index('--name')+1]
 containers[name] = {'State': {'Running': True, 'OOMKilled': False}, 'Config': {'Labels': {'com.nvidia.mission-control-showcase.run': name.removesuffix('-nova-carter-sil')}}}
 p.write_text(json.dumps(containers)); print('fixture-id')
elif args[0] == 'exec':
 print(json.dumps({'processes': {'nav': True}, 'oom_events': {'oom': 0, 'oom_kill': 0}}))
elif args[0] == 'stop':
 containers[args[1]]['State']['Running'] = False
 p.write_text(json.dumps(containers))
elif args[0] != 'logs':
 raise ValueError(args)
'''

FAKE_CLOUD = r'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
project=os.environ['COMPOSE_PROJECT_NAME']
names=['mosquitto','postgres'] if os.environ['TEST_SCENARIO']=='partial-cloud' else ['mosquitto','postgres','mission-database','mission-dispatch','wpg','cuopt','mission-control']
items={project+'-'+name+'-1': {'State': {'Running': True,'OOMKilled': False}, 'Config': {'Labels': {'com.docker.compose.project': project}}} for name in names}
(Path(os.environ['TEST_BIN'])/'containers.json').write_text(json.dumps(items))
sys.exit(8 if os.environ['TEST_SCENARIO']=='partial-cloud' else 0)
'''

FAKE_HELPER = r'''import copy,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/'package'/'scripts'))
import showcase
count=0
scenario=os.environ['TEST_SCENARIO']
robot={'name':'carter01','lifecycle':'ALIVE','status':{'online':True,'state':'IDLE','position_initialized':True,'errors':{},'pose':{'x':0.0,'y':0.0,'theta':0.0}}}
def fetch(url):
 global count
 if '/mission/' in url:
  state='COMPLETED' if sys.argv[1]=='finalize' or count>=2 else 'RUNNING'
  if scenario=='mission-failure' and count>=2: state='FAILED'
  return {'name':url.rsplit('/',1)[-1],'status':{'state':state}}
 count+=1
 robot['status']['pose']['x']=0.0 if count==1 else 1.0
 return copy.deepcopy(robot)
showcase.fetch_json=fetch
showcase.post_json=lambda *a,**k:'{}'
if sys.argv[1]=='wait-demo-complete':
 args=showcase.build_parser().parse_args()
 if scenario=='timeout': args.timeout=0
 if scenario=='isaac-exit':
  import json,signal,time
  manifest=json.loads((Path(sys.argv[sys.argv.index('--work-dir')+1])/'run-manifest.json').read_text())
  os.kill(manifest['isaac']['pid'],signal.SIGTERM)
  time.sleep(5)
 if scenario=='interrupt':
  (Path(os.environ['TEST_BIN'])/'waiting').touch()
  import time
  time.sleep(30)
 sys.exit(args.func(args))
sys.exit(showcase.main())
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / "package"  # Deliberately no ancestor named skills.
        shutil.copytree(PACKAGE_DIR, self.skill, ignore=shutil.ignore_patterns("__pycache__", "tests"))
        runner = self.skill / "scripts/run.sh"
        text = runner.read_text().replace('RUNNER_PID="$BASHPID"', BOUNDARIES+'\nRUNNER_PID="$BASHPID"')
        runner.write_text(text)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, text in [("docker", FAKE_DOCKER), ("cloud", FAKE_CLOUD), ("helper.py", FAKE_HELPER)]:
            p = self.bin / name
            p.write_text(text)
            p.chmod(0o755)
        isaac = self.root / "isaac"
        isaac.mkdir()
        (isaac / "isaac-sim.sh").write_text("#!/bin/bash\nexec sleep 300\n")
        (isaac / "isaac-sim.sh").chmod(0o755)
        self.work = self.root / "run"
        self.env = {
            'PATH': str(self.bin)+os.pathsep+os.environ['PATH'], 'HOME': str(self.root),
            'TMPDIR': str(self.root), 'TEST_BIN': str(self.bin), 'TEST_SCENARIO': 'success',
            'ISAAC_SIM_DIR': str(isaac), 'SHOWCASE_HELPER': str(self.bin/'helper.py'),
        }
        self.addCleanup(self.stop_child)

    def stop_child(self):
        isaac = run_state.load(self.work).get('isaac', {})
        if showcase.process_running(isaac):
            os.kill(int(isaac['pid']),signal.SIGTERM)

    def run_runner(self, *args):
        return subprocess.run(['bash',str(self.skill/'scripts/run.sh'),*args,
                               '--work-dir',str(self.work)],env=self.env,
                              capture_output=True,text=True,timeout=25)

    def test_success_and_idempotent_cleanup_preserve_acceptance(self):
        result=self.run_runner('--demo')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        artifact=run_state.result_path(self.work).read_bytes()
        manifest=run_state.load(self.work)
        self.assertEqual(manifest['state'],'accepted')
        self.assertEqual(len(manifest['containers']),8)
        self.assertTrue(manifest['mission_id'])
        self.assertTrue(manifest['motion']['motion_confirmed'])
        self.assertTrue(manifest['isaac']['start_ticks'])
        # Cleanup does not need the dependency doctor or scene restore script.
        (self.skill/'scripts/doctor.py').unlink()
        (self.skill/'scripts/restore_isaac_scene.py').unlink()
        for _ in range(2):
            stopped=self.run_runner('--stop')
            self.assertEqual(stopped.returncode,0,stopped.stdout+stopped.stderr)
        self.assertEqual(run_state.load(self.work)['state'],'stopped')
        self.assertTrue(run_state.load(self.work)['acceptance']['accepted'])
        self.assertEqual(run_state.result_path(self.work).read_bytes(),artifact)
        status = subprocess.run(
            [sys.executable, str(self.bin / "helper.py"), "run-status",
             "--work-dir", str(self.work), "--sample-pose"],
            env=self.env, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["motion"]["verdict"], "confirmed-historical")
        self.assertEqual(run_state.result_path(self.work).read_bytes(), artifact)
        self.assertTrue(all(not c['State']['Running'] for c in json.loads((self.bin/'containers.json').read_text()).values()))

    def test_stop_dry_run_and_unrelated_ownership(self):
        result = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        data_path = self.bin / 'containers.json'
        before = data_path.read_bytes()
        result = self.run_runner('--stop', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data_path.read_bytes(), before)
        containers = json.loads(before)
        first = next(iter(containers.values()))
        first['Config']['Labels'] = {'com.docker.compose.project': 'unrelated'}
        data_path.write_text(json.dumps(containers))
        result = self.run_runner('--stop')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(next(iter(json.loads(data_path.read_text()).values()))['State']['Running'])

    def test_ready_without_demo_is_not_mission_acceptance(self):
        result=self.run_runner()
        self.assertEqual(result.returncode,0,result.stderr)
        artifact=json.loads(run_state.result_path(self.work).read_text())
        self.assertEqual(artifact['state'],'ready')
        self.assertFalse(artifact['acceptance'].get('accepted',False))

    def test_failures_produce_results_with_original_status(self):
        for scenario,code,phase in [('preflight-failure',7,'preflight'),
                                    ('partial-cloud',8,'starting-cloud'),
                                    ('mission-failure',1,'mission-running'),
                                    ('timeout',1,'mission-running'),
                                    ('isaac-exit',1,'mission-running')]:
            with self.subTest(scenario=scenario):
                self.env['TEST_SCENARIO']=scenario
                self.work=self.root/scenario
                result=self.run_runner('--demo')
                self.assertEqual(result.returncode,code,result.stdout+result.stderr)
                artifact=json.loads(run_state.result_path(self.work).read_text())
                self.assertEqual(artifact['state'],'failed')
                self.assertIn(phase,artifact['acceptance']['reasons'][0])
                self.assertEqual(artifact['failure']['exit_code'],code)
                if scenario=='isaac-exit':
                    self.assertFalse(artifact['isaac']['running'])
                    self.assertEqual(artifact['isaac']['exit_status'],143)
                if scenario=='mission-failure':
                    self.assertTrue(artifact['motion']['motion_confirmed'])
                self.stop_child()
                stopped=self.run_runner('--stop')
                self.assertEqual(stopped.returncode,0,stopped.stderr)

    def test_interrupt_records_failure(self):
        self.env['TEST_SCENARIO']='interrupt'
        process=subprocess.Popen(['bash',str(self.skill/'scripts/run.sh'),'--demo',
                                  '--work-dir',str(self.work)],env=self.env,
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline=time.monotonic()+15
        while not (self.bin/'waiting').exists() and time.monotonic()<deadline:
            time.sleep(0.05)
        self.assertTrue((self.bin/'waiting').exists())
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10),143)
        artifact=json.loads(run_state.result_path(self.work).read_text())
        self.assertEqual(artifact['state'],'failed')
        self.assertIn('interrupted',artifact['acceptance']['reasons'][0])

    def test_preflight_and_dry_run_create_no_run_artifacts(self):
        for flag in ['--preflight','--dry-run']:
            with self.subTest(flag=flag):
                result=self.run_runner(flag)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertFalse(self.work.exists())

    def test_generated_env_does_not_export_legacy_entrypoints(self):
        manifest={'manifest_path':str(self.root/'manifest.json'), 'upstream_root':str(self.root),
                  'upstreams':{},'skills':{'bring-up-cloud-stack':{
                      'skill_dir':str(self.root/'upstream with spaces'),
                      'entrypoints':{'up':'unused'},'runner_env':{'up':'BRING_UP_CLOUD'}}}}
        env_file=self.root/'deps.env'
        env_file.write_text(doctor.env_lines(manifest))
        result=subprocess.run(['bash','-c','source "$1"; exec bash "$2" --dry-run --work-dir "$3"',
                               'test',str(env_file),str(self.skill/'scripts/run.sh'),str(self.work)],
                              env=self.env,text=True,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        legacy=self.run_runner('--preflight')
        self.assertEqual(legacy.returncode,0,legacy.stderr)
        self.env['BRING_UP_CLOUD']='direct-entrypoint'
        self.assertEqual(self.run_runner('--preflight').returncode,2)


if __name__ == '__main__':
    unittest.main()
