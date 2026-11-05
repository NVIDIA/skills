#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the documented run-config and broker setup blocks without services."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import yaml

REFERENCES = Path(__file__).resolve().parents[1] / 'references'


def embedded_python(name, marker):
    text = (REFERENCES / name).read_text()
    return text.split(f"python3 - <<'{marker}'\n", 1)[1].split(f'\n{marker}', 1)[0]


class TestGeneratedConfigs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'run'
        self.output.mkdir()
        self.template = self.root / 'shared-template.txt'
        self.template.write_text('[property]\nonnx-file=model.onnx\n')
        self.shared_cache = self.root / 'shared.engine'
        self.shared_cache.write_bytes(b'existing-cache')
        (self.output / 'config_deepstream.txt').write_text(
            '[primary-gie]\nmodel-engine-file=/workspace/models/model_b4_gpu0_fp16.engine\n'
            '[sink3]\nenable=1\nmsg-broker-conn-str=localhost;9092;mv3dt\n')
        (self.output / 'config_pgie.txt').write_text(self.template.read_text())
        (self.output / 'pub_sub_info_config_0.yml').write_text(yaml.safe_dump({
            'pubBrokerTopicStr': ['127.0.0.1:1883;/trck/cam1'],
            'subPeerBrokerTopicStrs': [['127.0.0.1:1883;/trck/cam2']],
        }))
        (self.output / 'config_mqtt.txt').write_text('[message-broker]\n')
        self.mqtt_config = self.root / 'mqtt.conf'
        self.mqtt_config.write_text('[message-broker]\nset-threaded=0\n')
        self.mqtt_config.chmod(0o600)
        self.env = {
            'PATH': os.defpath,
            'MV3DT_CONFIG_OUTPUT_DIR': str(self.output),
            'MV3DT_KAFKA_HOST': '172.17.0.1', 'MV3DT_KAFKA_PORT': '19092',
            'MV3DT_MQTT_HOST': '172.17.0.1', 'MV3DT_MQTT_PORT': '11883',
            'MV3DT_MQTT_CONFIG_FILE': str(self.mqtt_config),
            'MV3DT_ENABLE_STRONGLY_TYPED': '0',
        }

    def run_config(self, **overrides):
        return subprocess.run([sys.executable, '-'],
                              input=embedded_python('generate-configs.md', 'PYCONFIG'),
                              text=True, capture_output=True, env={**self.env, **overrides})

    def test_bridge_endpoints_and_private_adapter_config(self):
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('172.17.0.1;19092;mv3dt', (self.output / 'config_deepstream.txt').read_text())
        peer = yaml.safe_load((self.output / 'pub_sub_info_config_0.yml').read_text())
        self.assertEqual(peer['pubBrokerTopicStr'], ['172.17.0.1:11883;/trck/cam1'])
        self.assertEqual(peer['subPeerBrokerTopicStrs'], [['172.17.0.1:11883;/trck/cam2']])
        self.assertEqual((self.output / 'config_mqtt.txt').stat().st_mode & 0o777, 0o600)
        self.assertNotIn('strongly-typed=', (self.output / 'config_pgie.txt').read_text())

    def test_strongly_typed_changes_only_run_files(self):
        result = self.run_config(MV3DT_ENABLE_STRONGLY_TYPED='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.template.read_text(), '[property]\nonnx-file=model.onnx\n')
        self.assertEqual(self.shared_cache.read_bytes(), b'existing-cache')
        self.assertIn('strongly-typed=1', (self.output / 'config_pgie.txt').read_text())
        cache = '/workspace/experiments/engines/model_b4_gpu0_fp16-strongly-typed.engine'
        for name in ('config_deepstream.txt', 'config_pgie.txt'):
            self.assertIn(cache, (self.output / name).read_text())

    def test_invalid_bridge_endpoint_does_not_modify_configs(self):
        path = self.output / 'config_deepstream.txt'
        before = path.read_bytes()
        result = self.run_config(MV3DT_KAFKA_HOST='localhost')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_symlink_output_is_rejected(self):
        path = self.output / 'config_pgie.txt'
        path.unlink()
        path.symlink_to(self.template)
        result = self.run_config()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.template.read_text(), '[property]\nonnx-file=model.onnx\n')

    def test_world_readable_adapter_source_is_rejected(self):
        self.mqtt_config.chmod(0o644)
        result = self.run_config()
        self.assertNotEqual(result.returncode, 0)

    def test_all_peer_files_are_rewritten(self):
        peer = {'pubBrokerTopicStr': ['127.0.0.1:1883;/trck/cam5'],
                'subPeerBrokerTopicStrs': [['127.0.0.1:1883;/trck/cam1']]}
        path = self.output / 'pub_sub_info_config_1.yml'
        path.write_text(yaml.safe_dump(peer))
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = yaml.safe_load(path.read_text())
        self.assertEqual(updated['pubBrokerTopicStr'], ['172.17.0.1:11883;/trck/cam5'])

    def test_whitespace_in_generated_properties_does_not_create_duplicates(self):
        path = self.output / 'config_deepstream.txt'
        path.write_text(path.read_text().replace('msg-broker-conn-str=', '  msg-broker-conn-str = '))
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(path.read_text().count('msg-broker-conn-str'), 1)

    def test_missing_adapter_path_is_a_clear_failure(self):
        result = self.run_config(MV3DT_MQTT_CONFIG_FILE='')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MV3DT_MQTT_CONFIG_FILE', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


class TestKafkaBridgeListener(unittest.TestCase):
    def test_preserves_host_and_controller_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / 'config'
            config.mkdir()
            path = config / 'server.properties'
            original = ('listeners=PLAINTEXT://:9092,CONTROLLER://:9093\n'
                        'advertised.listeners=PLAINTEXT://localhost:9092\n'
                        'listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT\n'
                        'log.dirs=/existing/storage\n')
            path.write_text(original)
            env = {'PATH': os.defpath, 'KAFKA_DIR': temp_dir,
                   'MV3DT_KAFKA_HOST': '172.17.0.1', 'MV3DT_KAFKA_PORT': '19092'}
            script = embedded_python('container-network.md', 'PY')
            for _ in range(2):
                result = subprocess.run([sys.executable, '-'], input=script,
                                        text=True, capture_output=True, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
            updated = path.read_text()
            self.assertIn('PLAINTEXT://localhost:9092,MV3DT://172.17.0.1:19092', updated)
            self.assertIn('CONTROLLER://:9093', updated)
            self.assertIn('log.dirs=/existing/storage', updated)
            backups = list(config.glob('server.properties.backup-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)


class TestDocumentedLaunch(unittest.TestCase):
    def test_display_and_headless_use_bridge_and_scoped_driver_capabilities(self):
        stub = '''docker() {
  if [ "$1" = info ]; then printf '{"nvidia":{}}';
  elif [ "$1" = run ]; then printf 'LAUNCH:'; printf '<%s>' "$@";
  else return 1; fi
}
'''
        with tempfile.TemporaryDirectory() as temp_dir:
            for name in ('sample-run.md', 'custom-dataset.md'):
                text = (REFERENCES / name).read_text()
                launch = next(block for block in re.findall(r'^```bash\n(.*?)^```', text, re.M | re.S)
                              if 'MV3DT_DOCKER_LAUNCH_AT=' in block)
                for mode in ('display', 'headless'):
                    env = {'PATH': os.defpath, 'MODEL_REPO': temp_dir, 'DATASET_DIR': temp_dir,
                           'EXPERIMENT_DIR': temp_dir, 'DISPLAY': ':99', 'MV3DT_RUN_MODE': mode,
                           'MV3DT_KAFKA_HOST': '172.17.0.1', 'MV3DT_KAFKA_PORT': '19092'}
                    result = subprocess.run(['bash'], input=stub + launch, text=True,
                                            capture_output=True, env=env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('<--network><bridge>', result.stdout)
                    # Exact argv comparison rejects every unexpected launch option,
                    # including unsafe network, device, or permission additions.
                    capabilities = 'compute,utility,video'
                    if mode == 'display':
                        capabilities += ',graphics,display'
                    expected = ['run', '-t', '--rm', '--network', 'bridge', '--runtime=nvidia',
                                '-e', 'NVIDIA_DRIVER_CAPABILITIES=' + capabilities,
                                '-v', temp_dir + ':/workspace/models',
                                '-v', temp_dir + ':/workspace/inputs:ro',
                                '-v', temp_dir + ':/workspace/experiments']
                    if mode == 'display':
                        expected += ['-v', '/tmp/.X11-unix/:/tmp/.X11-unix', '-e', 'DISPLAY=:99']
                    expected += ['-w', '/workspace/experiments',
                                 'nvcr.io/nvidia/deepstream:9.1-triton-multiarch',
                                 'deepstream-test5-app', '-c', 'config_deepstream.txt']
                    self.assertEqual(re.findall(r'<([^>]*)>', result.stdout), expected)
                    self.assertIn('/workspace/inputs:ro', result.stdout)
                    self.assertEqual('/tmp/.X11-unix' in result.stdout, mode == 'display')
                    self.assertEqual(',graphics,display' in result.stdout, mode == 'display')


if __name__ == '__main__':
    unittest.main()
