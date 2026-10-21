from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preflight.py"
SPEC = importlib.util.spec_from_file_location("cdf_preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def toolkit_pod(
    name: str,
    node_name: str,
) -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": "gpu-operator",
            "labels": {"app": "nvidia-container-toolkit-daemonset"},
            "ownerReferences": [
                {
                    "kind": "DaemonSet",
                    "name": "nvidia-container-toolkit-daemonset",
                }
            ],
        },
        "spec": {
            "nodeName": node_name,
            "volumes": [
                {"name": "host-root", "hostPath": {"path": "/"}},
                {
                    "name": "driver-install-dir",
                    "hostPath": {"path": "/run/nvidia/driver"},
                },
            ],
            "containers": [
                {
                    "name": "nvidia-container-toolkit-ctr",
                    "volumeMounts": [
                        {
                            "name": "host-root",
                            "mountPath": "/host",
                            "readOnly": True,
                        },
                        {
                            "name": "driver-install-dir",
                            "mountPath": "/driver-root",
                        },
                    ],
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "nvidia-container-toolkit-ctr", "ready": True}],
        },
    }


def probe_pod(name="probe", node="gpu-a", gpu=True):
    return {
        "metadata": {"name": name, "namespace": "default"},
        "spec": {
            "nodeName": node,
            "containers": [
                {
                    "name": "check",
                    "resources": {"limits": {"nvidia.com/gpu": int(gpu)}},
                }
            ],
        },
        "status": {"phase": "Running", "containerStatuses": [{"name": "check", "ready": True}]},
    }


class PreflightUnitTests(unittest.TestCase):
    def test_runner_keeps_large_json_output_parseable(self):
        script = "import json; print(json.dumps({'items': ['x' * 100] * 30000}))"
        result = preflight.Runner().run([sys.executable, "-c", script])
        payload = preflight.json_object(result)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["items"]), 30000)

    def test_runner_reports_stdout_overflow_instead_of_truncating(self):
        script = "print('x' * 200)"
        with mock.patch.object(preflight, "MAX_COMMAND_STDOUT", 100):
            result = preflight.Runner().run([sys.executable, "-c", script])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("exceeded", result.stderr)

    def test_explicit_probe_api_results_and_command(self):
        for code, marker, expected in (
            (0, "REACHABLE", preflight.READY),
            (0, "UNREACHABLE", preflight.BLOCKED),
            (127, "python3 missing", preflight.UNVERIFIED),
            (1, "forbidden", preflight.UNVERIFIED),
        ):
            with self.subTest(marker=marker):
                runner = mock.Mock()
                runner.run.return_value = preflight.CommandResult(code, marker)
                check = preflight.kubernetes_api_endpoint_check(
                    runner,
                    ["kubectl", "--kubeconfig", "/tmp/selected"],
                    "https://api.example.test:6443",
                    {"gpu-a"},
                    {"items": [probe_pod()]},
                    ["default/probe/check"],
                )
                self.assertEqual(check.status, expected)
                self.assertTrue(check.required_for_install)
                self.assertEqual(
                    runner.run.call_args.args[0],
                    [
                        "kubectl",
                        "--kubeconfig",
                        "/tmp/selected",
                        "exec",
                        "-n",
                        "default",
                        "probe",
                        "-c",
                        "check",
                        "--",
                        "python3",
                        "-c",
                        preflight.PYTHON_API_PROBE,
                        "api.example.test",
                        "6443",
                    ],
                )
                self.assertEqual(runner.run.call_args.kwargs["timeout"], 10)

    def test_explicit_probe_rejects_invalid_targets_without_falling_back(self):
        for defect in (
            "hostNetwork",
            "notReady",
            "terminating",
            "pending",
            "missing",
            "wrongContainer",
        ):
            with self.subTest(defect=defect):
                pod = probe_pod()
                reference = "default/probe/check"
                if defect == "hostNetwork":
                    pod["spec"]["hostNetwork"] = True
                elif defect == "notReady":
                    pod["status"]["containerStatuses"][0]["ready"] = False
                elif defect == "terminating":
                    pod["metadata"]["deletionTimestamp"] = "now"
                elif defect == "pending":
                    pod["status"]["phase"] = "Pending"
                else:
                    reference = (
                        "default/missing/check" if defect == "missing" else "default/probe/sidecar"
                    )
                pods = {"items": [pod, toolkit_pod("toolkit", "gpu-a")]}
                runner = mock.Mock()
                check = preflight.kubernetes_api_endpoint_check(
                    runner,
                    ["kubectl"],
                    "https://api.example.test:6443",
                    {"gpu-a"},
                    pods,
                    [reference],
                )
                self.assertEqual(check.status, preflight.UNVERIFIED)
                check, ready = preflight.nvcuvid_library_check(
                    runner,
                    ["kubectl"],
                    {"gpu-a"},
                    {"gpu-a"},
                    pods,
                    [reference],
                )
                self.assertEqual(check.status, preflight.UNVERIFIED)
                self.assertIsNone(ready)
                runner.run.assert_not_called()
        runner = mock.Mock()
        check = preflight.kubernetes_api_endpoint_check(
            runner,
            ["kubectl"],
            "https://127.0.0.1:6443",
            {"gpu-a"},
            {"items": [probe_pod()]},
            ["default/probe/check"],
        )
        self.assertEqual(check.status, preflight.BLOCKED)
        runner.run.assert_not_called()

    def test_selected_gpu_probes_require_node_coverage_and_gpu_allocation(self):
        for gpu, second_node, marker, code, expected in (
            (True, True, "PRESENT", 0, preflight.READY),
            (True, False, "PRESENT", 0, preflight.UNVERIFIED),
            (False, True, "PRESENT", 0, preflight.UNVERIFIED),
            (True, True, "MISSING", 0, preflight.BLOCKED),
            (True, True, "PRESENT", 1, preflight.UNVERIFIED),
        ):
            with self.subTest(gpu=gpu, second_node=second_node, marker=marker, code=code):
                pods = [probe_pod(gpu=gpu)]
                references = ["default/probe/check"]
                if second_node:
                    pods.append(probe_pod("probe-b", "gpu-b", gpu=gpu))
                    references.append("default/probe-b/check")
                runner = mock.Mock()
                runner.run.return_value = preflight.CommandResult(code, marker)
                check, _ = preflight.nvcuvid_library_check(
                    runner,
                    ["kubectl"],
                    {"gpu-a", "gpu-b"},
                    {"gpu-a", "gpu-b"},
                    {"items": pods},
                    references,
                )
                self.assertEqual(check.status, expected)
                for call in runner.run.call_args_list:
                    self.assertEqual(
                        call.args[0][-3:], ["python3", "-c", preflight.PYTHON_NVCUVID_PROBE]
                    )
                if not gpu:
                    runner.run.assert_not_called()

    def test_build_environment_and_scoped_push_confirmation(self):
        repository = "nvcr.io/team/controller"
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "make").mkdir()
            (repo / "make/common.mk").write_text(
                "AIRFLOW_IMAGE_REPOSITORY ?= nvcr.io/old/controller\nAIRFLOW_IMAGE_TAG ?= latest\n"
            )
            for configured, tag, confirmed, expected in (
                (repository, "test-1", repository, preflight.READY),
                (repository, "test-1", None, preflight.UNVERIFIED),
                (repository, "test-1", "nvcr.io/other/controller", preflight.UNVERIFIED),
                ("", "test-1", repository, preflight.BLOCKED),
                (repository, "", repository, preflight.BLOCKED),
            ):
                with self.subTest(configured=configured, tag=tag, confirmed=confirmed):
                    with mock.patch.dict(
                        preflight.os.environ,
                        {
                            "AIRFLOW_IMAGE_REPOSITORY": configured,
                            "AIRFLOW_IMAGE_TAG": tag,
                        },
                    ):
                        runner = mock.Mock()
                        checks = {
                            c.id: c
                            for c in preflight.repository_checks(
                                repo,
                                {},
                                runner,
                                confirmed_push_repository=confirmed,
                            )
                        }
                    self.assertEqual(checks["image.repository"].status, expected)
                    self.assertTrue(checks["image.repository"].required_for_install)
                    runner.run.assert_not_called()
                    if expected == preflight.READY:
                        self.assertIn("User-confirmed", checks["image.repository"].summary)
                        self.assertIn("test-1", checks["image.tag"].summary)

    def test_new_cli_options_reach_audit_and_do_not_bypass_prebuilt(self):
        with mock.patch.object(preflight, "build_report", return_value={}) as report:
            with mock.patch("builtins.print"):
                preflight.main(
                    [
                        "audit",
                        "--format",
                        "json",
                        "--probe-pod",
                        "default/a/check",
                        "--probe-pod",
                        "default/b/check",
                        "--confirmed-push-repository",
                        "nvcr.io/team/image",
                    ]
                )
            self.assertEqual(
                report.call_args.kwargs["probe_pods"], ["default/a/check", "default/b/check"]
            )
            self.assertEqual(
                report.call_args.kwargs["confirmed_push_repository"], "nvcr.io/team/image"
            )
        runner = mock.Mock()
        runner.run.return_value = preflight.CommandResult(1, "", "denied")
        checks = {
            c.id: c
            for c in preflight.repository_checks(
                Path("/nonexistent-cdf-repo"),
                {},
                runner,
                prebuilt_controller_image="nvcr.io/team/image:v1",
                confirmed_push_repository="nvcr.io/team/image",
            )
        }
        self.assertEqual(checks["image.pull_access"].status, preflight.UNVERIFIED)

    def test_prebuilt_image_reference_splits_repository_and_tag(self) -> None:
        self.assertEqual(
            preflight.parse_prebuilt_controller_image(
                "nvcr.io/example-org/example-team/sdg-workflow-airflow:release-1"
            ),
            (
                "nvcr.io/example-org/example-team/sdg-workflow-airflow",
                "release-1",
            ),
        )
        self.assertEqual(
            preflight.parse_prebuilt_controller_image(
                "registry.example.com:5000/team/controller:v1"
            ),
            ("registry.example.com:5000/team/controller", "v1"),
        )
        for invalid in (
            "controller:v1",
            "nvcr.io/team/controller",
            "nvcr.io/team/controller@sha256:deadbeef",
            "https://nvcr.io/team/controller:v1",
            "nvcr.io/team/controller:bad tag",
            "NVCR.IO/team/controller:v1",
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(preflight.parse_prebuilt_controller_image(invalid))

    def test_prebuilt_image_uses_manifest_pull_access_as_install_gate(self) -> None:
        class ManifestRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                self.assert_manifest_command = command
                return preflight.CommandResult(0, '{"schemaVersion": 2}')

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for path in (
                "Makefile",
                "make/setup.mk",
                "make/install.mk",
                "deploy/values.yaml",
                "secrets.env.example",
                "scripts/install_nfs.sh",
                "secrets.env",
            ):
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            checks = preflight.repository_checks(
                repo,
                {
                    "NGC_API_KEY": True,
                    "HF_TOKEN": True,
                    "AWS_S3_BUCKET": True,
                    "AWS_S3_REGION": True,
                    "AWS_S3_ACCESS_KEY_ID": True,
                    "AWS_S3_SECRET_ACCESS_KEY": True,
                },
                ManifestRunner(),
                prebuilt_controller_image="nvcr.io/team/controller:v1",
            )
        by_id = {check.id: check for check in checks}
        self.assertEqual(by_id["image.repository"].status, preflight.READY)
        self.assertEqual(by_id["image.pull_access"].status, preflight.READY)
        self.assertTrue(by_id["image.pull_access"].required_for_install)
        self.assertEqual(preflight.install_status(checks), preflight.READY)

    def test_prebuilt_image_is_unverified_without_manifest_access(self) -> None:
        class AuthenticationRequiredRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                return preflight.CommandResult(1, "", "authentication required")

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for path in (
                "Makefile",
                "make/setup.mk",
                "make/install.mk",
                "deploy/values.yaml",
                "secrets.env.example",
                "scripts/install_nfs.sh",
                "secrets.env",
            ):
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            checks = preflight.repository_checks(
                repo,
                {
                    "NGC_API_KEY": True,
                    "HF_TOKEN": True,
                    "AWS_S3_BUCKET": True,
                    "AWS_S3_REGION": True,
                    "AWS_S3_ACCESS_KEY_ID": True,
                    "AWS_S3_SECRET_ACCESS_KEY": True,
                },
                AuthenticationRequiredRunner(),
                prebuilt_controller_image="nvcr.io/team/controller:v1",
            )
        pull_check = next(check for check in checks if check.id == "image.pull_access")
        self.assertEqual(pull_check.status, preflight.UNVERIFIED)
        self.assertNotIn("authentication required", pull_check.summary)
        self.assertEqual(preflight.install_status(checks), preflight.UNVERIFIED)

    def test_default_build_path_still_requires_nondefault_push_repository(self) -> None:
        class NoRun:
            def run(self, command: list[str], timeout: int = 20) -> None:
                raise AssertionError("default repository checks must not invoke a command")

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for path in (
                "Makefile",
                "make/setup.mk",
                "make/install.mk",
                "deploy/values.yaml",
                "secrets.env.example",
                "scripts/install_nfs.sh",
                "secrets.env",
            ):
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            (repo / "make/common.mk").write_text(
                "AIRFLOW_IMAGE_REPOSITORY ?= "
                "nvcr.io/<your-org>/<your-team>/sdg-workflow-airflow\n"
                "AIRFLOW_IMAGE_TAG ?= release-1\n",
                encoding="utf-8",
            )
            checks = preflight.repository_checks(
                repo,
                {
                    "NGC_API_KEY": True,
                    "HF_TOKEN": True,
                    "AWS_S3_BUCKET": True,
                    "AWS_S3_REGION": True,
                    "AWS_S3_ACCESS_KEY_ID": True,
                    "AWS_S3_SECRET_ACCESS_KEY": True,
                },
                NoRun(),
            )
        repository_check = next(check for check in checks if check.id == "image.repository")
        self.assertEqual(repository_check.status, preflight.BLOCKED)
        self.assertIn("make/common.mk", repository_check.summary)

    def test_image_settings_edited_in_common_mk_need_no_exports(self) -> None:
        repository = "nvcr.io/example-org/example-team/sdg-workflow-airflow"
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "make").mkdir()
            (repo / "make/common.mk").write_text(
                f"AIRFLOW_IMAGE_REPOSITORY ?= {repository}\nAIRFLOW_IMAGE_TAG ?= release-1\n",
                encoding="utf-8",
            )
            environment = {
                key: value
                for key, value in preflight.os.environ.items()
                if key not in ("AIRFLOW_IMAGE_REPOSITORY", "AIRFLOW_IMAGE_TAG")
            }
            for confirmed, expected in (
                (repository, preflight.READY),
                (None, preflight.UNVERIFIED),
            ):
                with self.subTest(confirmed=confirmed):
                    with mock.patch.dict(preflight.os.environ, environment, clear=True):
                        checks = {
                            c.id: c
                            for c in preflight.repository_checks(
                                repo, {}, mock.Mock(), confirmed_push_repository=confirmed
                            )
                        }
                    self.assertEqual(checks["image.repository"].status, expected)
                    self.assertEqual(checks["image.tag"].status, preflight.READY)

    def test_prebuilt_mode_skips_docker_build_version_and_daemon_checks(self) -> None:
        class ToolRunner:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                self.commands.append(command)
                outputs = {
                    "python3": "Python 3.11.16",
                    "kubectl": '{"clientVersion":{"gitVersion":"v1.37.1"}}',
                    "helm": "v3.22.0",
                    "aws": "aws-cli/2.37.5 Python/3.13.11",
                }
                return preflight.CommandResult(0, outputs[command[0]])

        runner = ToolRunner()
        with mock.patch.object(preflight.shutil, "which", return_value="/bin/tool"):
            checks = preflight.tool_checks(runner, controller_build_required=False)
        by_id = {check.id: check for check in checks}
        self.assertEqual(by_id["tool.python"].status, preflight.READY)
        self.assertEqual(by_id["tool.docker"].status, preflight.READY)
        self.assertEqual(by_id["docker.daemon"].status, preflight.WARNING)
        self.assertFalse(any(command[0] == "docker" for command in runner.commands))

    def test_prebuilt_mode_blocks_without_docker_cli(self) -> None:
        class ToolRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                outputs = {
                    "python3": "Python 3.11.16",
                    "kubectl": '{"clientVersion":{"gitVersion":"v1.37.1"}}',
                    "helm": "v3.22.0",
                    "aws": "aws-cli/2.37.5 Python/3.13.11",
                }
                return preflight.CommandResult(0, outputs[command[0]])

        def which(executable: str) -> str | None:
            return None if executable == "docker" else f"/bin/{executable}"

        with mock.patch.object(preflight.shutil, "which", side_effect=which):
            checks = preflight.tool_checks(ToolRunner(), controller_build_required=False)
        docker_check = next(check for check in checks if check.id == "tool.docker")
        self.assertEqual(docker_check.status, preflight.BLOCKED)
        self.assertTrue(docker_check.required_for_install)

    def test_python_310_is_untested_warning_not_blocker(self) -> None:
        class ToolRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                outputs = {
                    "python3": "Python 3.10.12",
                    "kubectl": '{"clientVersion":{"gitVersion":"v1.37.1"}}',
                    "helm": "v3.22.0",
                    "docker": "Docker version 29.8.1",
                    "aws": "aws-cli/2.37.5 Python/3.13.11",
                }
                return preflight.CommandResult(0, outputs[command[0]])

        with mock.patch.object(preflight.shutil, "which", return_value="/bin/tool"):
            checks = preflight.tool_checks(ToolRunner(), controller_build_required=False)
        python_check = next(check for check in checks if check.id == "tool.python")
        self.assertEqual(python_check.status, preflight.WARNING)
        self.assertIn("Python 3.11 recommended", python_check.summary)
        self.assertIn("untested", python_check.summary)

    def test_untested_tool_version_warns_without_blocking_install(self) -> None:
        warning = preflight.Check("tool.python", "tools", preflight.WARNING, "untested", True)
        self.assertEqual(preflight.install_status([warning]), preflight.READY)

    def test_prebuilt_parser_and_quick_start_omit_build(self) -> None:
        args = preflight.build_parser().parse_args(
            [
                "audit",
                "--prebuilt-controller-image",
                "nvcr.io/team/controller:v1",
            ]
        )
        self.assertEqual(args.prebuilt_controller_image, "nvcr.io/team/controller:v1")
        rendered = preflight.render_text(
            {
                "install_readiness": preflight.READY,
                "run_readiness": preflight.BLOCKED,
                "checks": [],
                "workflows": [],
                "prebuilt_controller_image": "nvcr.io/team/controller:v1",
            }
        )
        self.assertNotIn("make build sdg-controller", rendered)
        self.assertIn("defaultAirflowRepository=nvcr.io/team/controller", rendered)
        self.assertIn("defaultAirflowTag=v1", rendered)

    def test_invalid_prebuilt_reference_never_renders_build_instructions(self) -> None:
        rendered = preflight.render_text(
            {
                "install_readiness": preflight.BLOCKED,
                "run_readiness": preflight.BLOCKED,
                "checks": [],
                "workflows": [],
                "prebuilt_controller_image": "nvcr.io/team/controller",
            }
        )
        self.assertIn("Correct --prebuilt-controller-image", rendered)
        self.assertNotIn("make build sdg-controller", rendered)
        self.assertNotIn("make install sdg-controller", rendered)

    def test_profile_catalog_matches_packaged_dags_and_pools(self) -> None:
        skill_root = Path(__file__).resolve().parents[1]
        repo_root = Path(__file__).resolve().parents[3]
        payload = json.loads(
            (skill_root / "config/readiness-profiles.json").read_text(encoding="utf-8")
        )
        profiles = payload["profiles"]
        expected = {
            "image_attribute_augmentation_dag_k8s",
            "event_video_generation_dag_k8s",
            "event_video_generation_dag_osmo",
            "video_data_augmentation_dag_k8s",
            "video_data_augmentation_dag_osmo",
            "defect_image_generation_smt_dag_k8s",
            "autonomous_vehicle_dataset_generation_dag_k8s",
            "k8s_resource_cleanup_dag",
        }
        self.assertEqual({item["dag_id"] for item in profiles}, expected)
        values = (repo_root / "deploy/values.yaml").read_text(encoding="utf-8")
        for profile in profiles:
            paths = [profile["package_path"], *profile.get("legacy_package_paths", [])]
            self.assertTrue(any((repo_root / path).is_file() for path in paths))
            for mode in profile["modes"]:
                for pool in mode["pools"]:
                    if pool != "default_pool":
                        self.assertIn(f"    {pool}:", values)
        self.assertEqual(
            {
                (profile["dag_id"], mode["name"])
                for profile in profiles
                for mode in profile["modes"]
                if mode.get("requires_nvcuvid")
            },
            {
                ("event_video_generation_dag_k8s", "external"),
                ("event_video_generation_dag_k8s", "internal"),
                ("video_data_augmentation_dag_k8s", "external"),
                ("video_data_augmentation_dag_k8s", "internal"),
            },
        )
        manifest = (
            repo_root
            / "airflow/dags/workflows/autonomous_vehicle_dataset_generation_dag/configs"
            / "autonomous_vehicle_dataset_generation_k8s_manifest.yaml"
        ).read_text(encoding="utf-8")
        av = next(
            item
            for item in profiles
            if item["dag_id"] == "autonomous_vehicle_dataset_generation_dag_k8s"
        )
        self.assertEqual(
            {mode["name"]: mode["gpu_floor"] for mode in av["modes"]},
            {"retrieval": 1, "generation": 4},
        )
        for mode in av["modes"]:
            self.assertFalse(mode.get("external_endpoints", False))
            claims = {claim["name"] for claim in mode["required_pvcs"]}
            self.assertEqual(claims, {"shared-ffmpeg"})
            for claim in claims:
                self.assertIn(f"claimName: {claim}", manifest)

    def test_parse_env_presence_never_evaluates_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            env_file = Path(directory) / "secrets.env"
            env_file.write_text(
                "NGC_API_KEY=$(touch " + str(marker) + ")\nHF_TOKEN=''\nAWS_S3_BUCKET=example\n",
                encoding="utf-8",
            )
            result = preflight.parse_env_presence(env_file, {})
            self.assertTrue(result["NGC_API_KEY"])
            self.assertFalse(result["HF_TOKEN"])
            self.assertTrue(result["AWS_S3_BUCKET"])
            self.assertFalse(marker.exists())

    def test_environment_overrides_empty_file_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "secrets.env"
            env_file.write_text("HF_TOKEN=\n", encoding="utf-8")
            result = preflight.parse_env_presence(env_file, {"HF_TOKEN": "secret-value"})
            self.assertTrue(result["HF_TOKEN"])

    def test_extract_storage_classes_finds_all_chart_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = Path(directory) / "values.yaml"
            values.write_text(
                'storageClass: nfs\nstorageClassName: "rwx-fast" # comment\nother: value\n',
                encoding="utf-8",
            )
            self.assertEqual(preflight.extract_storage_classes(values), {"nfs", "rwx-fast"})

    def test_repo_nfs_class_requires_available_provisioner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "deploy").mkdir()
            (repo / "deploy/values.yaml").write_text("storageClassName: nfs\n", encoding="utf-8")
            facts = preflight.KubernetesFacts(
                reachable=True,
                storage_classes={"nfs": "cluster.local/nfs-subdir-external-provisioner"},
                repo_nfs_provisioner_ready=False,
            )
            checks = preflight.storage_checks(repo, facts)
            self.assertEqual(checks[-1].status, preflight.BLOCKED)

    def test_kubeconfig_path_list_is_rejected_without_running_kubectl(self) -> None:
        class NoRun:
            def run(self, command: list[str], timeout: int = 20) -> None:
                self.fail("runner must not be called")

        checks, facts = preflight.kubernetes_audit(NoRun(), Path("one:two"))
        self.assertEqual(checks[0].status, preflight.BLOCKED)
        self.assertFalse(facts.reachable)

    def test_gpu_requests_use_scheduler_effective_request(self) -> None:
        pod = {
            "spec": {
                "initContainers": [{"resources": {"requests": {"nvidia.com/gpu": "3"}}}],
                "containers": [
                    {"resources": {"requests": {"nvidia.com/gpu": "1"}}},
                    {"resources": {"limits": {"nvidia.com/gpu": "1"}}},
                ],
            }
        }
        self.assertEqual(preflight.effective_gpu_requests(pod), 3)

    def test_node_inventory_tracks_every_gpu_node_and_ready_capacity(self) -> None:
        def node(name: str, ready: bool, allocatable_gpus: int, capacity_gpus: int = 0) -> dict:
            return {
                "metadata": {"name": name},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
                    "allocatable": {"nvidia.com/gpu": str(allocatable_gpus)},
                    "capacity": {"nvidia.com/gpu": str(capacity_gpus)},
                },
            }

        inventory = preflight.kubernetes_node_inventory(
            {
                "items": [
                    node("cpu-ready", True, 0),
                    node("gpu-ready-a", True, 2),
                    node("gpu-ready-b", True, 1),
                    node("gpu-not-ready", False, 4),
                    node("gpu-capacity-only", True, 0, 1),
                ]
            }
        )
        self.assertEqual(inventory[0], 4)
        self.assertEqual(inventory[1], 3)
        self.assertEqual(
            inventory[2],
            {
                "gpu-ready-a",
                "gpu-ready-b",
                "gpu-not-ready",
                "gpu-capacity-only",
            },
        )
        self.assertEqual(inventory[3], {"gpu-ready-a", "gpu-ready-b", "gpu-capacity-only"})

    def test_nvcuvid_check_inspects_every_ready_gpu_node(self) -> None:
        class ProbeRunner:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                self.commands.append(command)
                return preflight.CommandResult(0, "PRESENT\n")

        runner = ProbeRunner()
        check, ready = preflight.nvcuvid_library_check(
            runner,
            ["kubectl", "--kubeconfig", "/tmp/config"],
            {"gpu-a", "gpu-b"},
            {"gpu-a", "gpu-b"},
            {
                "items": [
                    toolkit_pod("toolkit-a", "gpu-a"),
                    toolkit_pod("toolkit-b", "gpu-b"),
                ]
            },
        )
        self.assertEqual(check.status, preflight.READY)
        self.assertTrue(ready)
        self.assertEqual(len(runner.commands), 2)
        self.assertEqual(
            {command[command.index("exec") + 3] for command in runner.commands},
            {"toolkit-a", "toolkit-b"},
        )

    def test_nvcuvid_check_blocks_on_missing_library(self) -> None:
        class ProbeRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                pod_name = command[command.index("exec") + 3]
                marker = "MISSING\n" if pod_name == "toolkit-b" else "PRESENT\n"
                return preflight.CommandResult(0, marker)

        check, ready = preflight.nvcuvid_library_check(
            ProbeRunner(),
            ["kubectl"],
            {"gpu-a", "gpu-b", "gpu-not-ready"},
            {"gpu-a", "gpu-b"},
            {
                "items": [
                    toolkit_pod("toolkit-a", "gpu-a"),
                    toolkit_pod("toolkit-b", "gpu-b"),
                ]
            },
        )
        self.assertEqual(check.status, preflight.BLOCKED)
        self.assertFalse(ready)
        self.assertIn("gpu-b", check.summary)
        self.assertIn("gpu-not-ready", check.summary)

    def test_nvcuvid_check_is_unverified_when_exec_fails(self) -> None:
        class FailingRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                return preflight.CommandResult(1, "", "forbidden")

        check, ready = preflight.nvcuvid_library_check(
            FailingRunner(),
            ["kubectl"],
            {"gpu-a"},
            {"gpu-a"},
            {"items": [toolkit_pod("toolkit-a", "gpu-a")]},
        )
        self.assertEqual(check.status, preflight.UNVERIFIED)
        self.assertIsNone(ready)
        self.assertIn("gpu-a", check.summary)

    def test_nvcuvid_probe_resolves_absolute_symlinks_in_each_mount_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for mount_name in ("host", "driver-root"):
                with self.subTest(mount_name=mount_name):
                    mount_root = Path(directory) / mount_name
                    library_dir = mount_root / "usr/lib/x86_64-linux-gnu"
                    target = mount_root / "opt/nvidia/libnvcuvid-real.so"
                    library_dir.mkdir(parents=True)
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b"library")
                    (library_dir / "libnvcuvid.so.1").symlink_to("/opt/nvidia/libnvcuvid-real.so")

                    result = subprocess.run(
                        [
                            "sh",
                            "-c",
                            preflight.NVCUVID_PROBE_SCRIPT,
                            "nvcuvid-probe",
                            f"{mount_root}:{mount_root / 'usr/lib'}",
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(result.stdout.strip(), "PRESENT")

    def test_nvcuvid_check_tries_all_rollout_candidates(self) -> None:
        class RolloutRunner:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                self.commands.append(command)
                pod_name = command[command.index("exec") + 3]
                marker = "ERROR\n" if pod_name == "toolkit-a" else "PRESENT\n"
                return preflight.CommandResult(0, marker)

        runner = RolloutRunner()
        check, ready = preflight.nvcuvid_library_check(
            runner,
            ["kubectl"],
            {"gpu-a"},
            {"gpu-a"},
            {
                "items": [
                    toolkit_pod("toolkit-a", "gpu-a"),
                    toolkit_pod("toolkit-b", "gpu-a"),
                ]
            },
        )
        self.assertEqual(check.status, preflight.READY)
        self.assertTrue(ready)
        self.assertEqual(len(runner.commands), 2)

    def test_toolkit_target_requires_gpu_operator_host_mount(self) -> None:
        lookalike = toolkit_pod("lookalike", "gpu-a")
        lookalike["metadata"].pop("ownerReferences")
        self.assertEqual(
            preflight.nvidia_toolkit_probe_targets({"items": [lookalike]}, {"gpu-a"}),
            {},
        )

    def test_api_endpoint_check_is_ready_when_pod_can_connect(self) -> None:
        class ReachableRunner:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                self.commands.append(command)
                self.timeout = timeout
                return preflight.CommandResult(0, "REACHABLE\n")

        runner = ReachableRunner()
        check = preflight.kubernetes_api_endpoint_check(
            runner,
            ["kubectl", "--kubeconfig", "/tmp/config"],
            "https://api.example.test:6443",
            {"gpu-a"},
            {"items": [toolkit_pod("toolkit-a", "gpu-a")]},
        )
        self.assertEqual(check.status, preflight.READY)
        self.assertEqual(runner.timeout, 10)
        self.assertEqual(
            runner.commands,
            [
                [
                    "kubectl",
                    "--kubeconfig",
                    "/tmp/config",
                    "exec",
                    "-n",
                    "gpu-operator",
                    "toolkit-a",
                    "-c",
                    "nvidia-container-toolkit-ctr",
                    "--",
                    "/busybox/sh",
                    "-c",
                    preflight.KUBERNETES_API_PROBE_SCRIPT,
                    "api-endpoint-probe",
                    "api.example.test",
                    "6443",
                ]
            ],
        )

    def test_api_endpoint_check_blocks_on_conclusive_connection_failure(self) -> None:
        class UnreachableRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                return preflight.CommandResult(0, "UNREACHABLE\n")

        check = preflight.kubernetes_api_endpoint_check(
            UnreachableRunner(),
            ["kubectl"],
            "https://api.example.test:6443",
            {"gpu-a"},
            {"items": [toolkit_pod("toolkit-a", "gpu-a")]},
        )
        self.assertEqual(check.status, preflight.BLOCKED)

    def test_api_endpoint_check_blocks_loopback_without_exec(self) -> None:
        class NoRun:
            def run(self, command: list[str], timeout: int = 20) -> None:
                self.fail("loopback endpoint must not be probed")

        for server in (
            "https://localhost:6443",
            "https://127.0.0.1:6443",
            "https://[::1]:6443",
        ):
            with self.subTest(server=server):
                check = preflight.kubernetes_api_endpoint_check(
                    NoRun(), ["kubectl"], server, {"gpu-a"}, {"items": []}
                )
                self.assertEqual(check.status, preflight.BLOCKED)

    def test_api_endpoint_check_is_unverified_without_probe_pod(self) -> None:
        class NoRun:
            def run(self, command: list[str], timeout: int = 20) -> None:
                self.fail("missing probe pod must not execute a command")

        check = preflight.kubernetes_api_endpoint_check(
            NoRun(),
            ["kubectl"],
            "https://api.example.test:6443",
            {"gpu-a"},
            {"items": []},
        )
        self.assertEqual(check.status, preflight.UNVERIFIED)

    def test_api_endpoint_check_is_unverified_when_exec_is_denied(self) -> None:
        class DeniedRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                return preflight.CommandResult(1, "", "Error from server (Forbidden)")

        check = preflight.kubernetes_api_endpoint_check(
            DeniedRunner(),
            ["kubectl"],
            "https://api.example.test:6443",
            {"gpu-a"},
            {"items": [toolkit_pod("toolkit-a", "gpu-a")]},
        )
        self.assertEqual(check.status, preflight.UNVERIFIED)

    def test_api_endpoint_check_is_unverified_without_busybox_nc(self) -> None:
        class MissingToolRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                return preflight.CommandResult(0, "MISSING_TOOL\n")

        check = preflight.kubernetes_api_endpoint_check(
            MissingToolRunner(),
            ["kubectl"],
            "https://api.example.test:6443",
            {"gpu-a"},
            {"items": [toolkit_pod("toolkit-a", "gpu-a")]},
        )
        self.assertEqual(check.status, preflight.UNVERIFIED)

    def test_api_endpoint_check_is_unverified_for_unsupported_nc_options(self) -> None:
        class UnsupportedToolRunner:
            def run(self, command: list[str], timeout: int = 20) -> preflight.CommandResult:
                return preflight.CommandResult(0, "UNSUPPORTED\n")

        check = preflight.kubernetes_api_endpoint_check(
            UnsupportedToolRunner(),
            ["kubectl"],
            "https://api.example.test:6443",
            {"gpu-a"},
            {"items": [toolkit_pod("toolkit-a", "gpu-a")]},
        )
        self.assertEqual(check.status, preflight.UNVERIFIED)

    def test_nvcuvid_check_without_gpu_nodes_is_unverified(self) -> None:
        class NoRun:
            def run(self, command: list[str], timeout: int = 20) -> None:
                self.fail("runner must not be called")

        check, ready = preflight.nvcuvid_library_check(
            NoRun(), ["kubectl"], set(), set(), {"items": []}
        )
        self.assertEqual(check.status, preflight.UNVERIFIED)
        self.assertIsNone(ready)

    def test_mode_blocks_on_known_gpu_shortfall(self) -> None:
        profile = {"dag_id": "example", "platform": "kubernetes"}
        mode = {"name": "internal", "gpu_floor": 4, "gpu_practical": 5, "pools": []}
        kube = preflight.KubernetesFacts(reachable=True, supported=True, free_gpus=2)
        controller = preflight.ControllerFacts(installed=True, healthy=True)
        result = preflight.mode_status(profile, mode, True, {}, kube, controller)
        self.assertEqual(result["status"], preflight.BLOCKED)
        self.assertTrue(any("needs at least 4" in item for item in result["blockers"]))

    def test_mode_blocks_when_nvcuvid_is_missing(self) -> None:
        profile = {"dag_id": "example", "platform": "kubernetes"}
        mode = {
            "name": "internal",
            "gpu_floor": 1,
            "pools": [],
            "requires_nvcuvid": True,
        }
        kube = preflight.KubernetesFacts(
            reachable=True,
            supported=True,
            free_gpus=1,
            nvcuvid_ready=False,
        )
        controller = preflight.ControllerFacts(
            installed=True,
            healthy=True,
            api_verified=True,
            dags={"example": {"dag_id": "example", "is_paused": False}},
        )
        result = preflight.mode_status(profile, mode, True, {}, kube, controller)
        self.assertEqual(result["status"], preflight.BLOCKED)
        self.assertTrue(any("libnvcuvid.so.1" in item for item in result["blockers"]))

    def test_mode_is_unverified_when_nvcuvid_cannot_be_inspected(self) -> None:
        profile = {"dag_id": "example", "platform": "kubernetes"}
        mode = {
            "name": "internal",
            "gpu_floor": 1,
            "pools": [],
            "requires_nvcuvid": True,
        }
        kube = preflight.KubernetesFacts(
            reachable=True,
            supported=True,
            free_gpus=1,
        )
        controller = preflight.ControllerFacts(
            installed=True,
            healthy=True,
            api_verified=True,
            dags={"example": {"dag_id": "example", "is_paused": False}},
        )
        result = preflight.mode_status(profile, mode, True, {}, kube, controller)
        self.assertEqual(result["status"], preflight.UNVERIFIED)
        self.assertTrue(any("libnvcuvid.so.1" in item for item in result["unverified"]))

    def _pvc_mode_status(self, pvc_phases: dict[str, str] | None) -> dict:
        profile = {"dag_id": "example", "platform": "kubernetes"}
        mode = {
            "name": "internal",
            "gpu_floor": 0,
            "pools": [],
            "required_pvcs": [
                {
                    "name": "shared-ffmpeg",
                    "remediation": "fill it as described in docs/av.md#ffmpeg",
                }
            ],
        }
        kube = preflight.KubernetesFacts(reachable=True, supported=True, free_gpus=0)
        controller = preflight.ControllerFacts(
            installed=True,
            healthy=True,
            api_verified=True,
            dags={"example": {"dag_id": "example", "is_paused": False}},
            pvc_phases=pvc_phases,
        )
        return preflight.mode_status(profile, mode, True, {}, kube, controller)

    def test_mode_blocks_when_required_pvc_is_missing(self) -> None:
        result = self._pvc_mode_status({"ngc-model-cache": "Bound"})
        self.assertEqual(result["status"], preflight.BLOCKED)
        self.assertTrue(
            any(
                "shared-ffmpeg is missing; fill it as described in docs/av.md#ffmpeg" in item
                for item in result["blockers"]
            )
        )

    def test_mode_blocks_when_required_pvc_is_not_bound(self) -> None:
        result = self._pvc_mode_status({"shared-ffmpeg": "Pending"})
        self.assertEqual(result["status"], preflight.BLOCKED)
        self.assertTrue(any("is Pending, not Bound" in item for item in result["blockers"]))

    def test_mode_is_ready_when_required_pvc_is_bound(self) -> None:
        result = self._pvc_mode_status({"shared-ffmpeg": "Bound"})
        self.assertEqual(result["status"], preflight.READY)

    def test_mode_is_unverified_when_pvcs_were_not_inspected(self) -> None:
        result = self._pvc_mode_status(None)
        self.assertEqual(result["status"], preflight.UNVERIFIED)
        self.assertTrue(any("shared-ffmpeg" in item for item in result["unverified"]))

    def test_external_mode_is_unverified_without_endpoint_evidence(self) -> None:
        profile = {"dag_id": "example", "platform": "kubernetes"}
        mode = {
            "name": "external",
            "gpu_floor": 0,
            "gpu_practical": 0,
            "pools": [],
            "external_endpoints": True,
        }
        kube = preflight.KubernetesFacts(
            reachable=True,
            supported=True,
            free_gpus=8,
            nvcuvid_ready=False,
        )
        controller = preflight.ControllerFacts(
            installed=True,
            healthy=True,
            api_verified=True,
            dags={"example": {"dag_id": "example", "is_paused": False}},
        )
        result = preflight.mode_status(profile, mode, True, {}, kube, controller)
        self.assertEqual(result["status"], preflight.UNVERIFIED)
        self.assertFalse(any("libnvcuvid.so.1" in item for item in result["unverified"]))

    def test_workflow_aggregation_reports_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "ready.yaml").write_text("x", encoding="utf-8")
            (repo / "blocked.yaml").write_text("x", encoding="utf-8")
            profiles = [
                {
                    "dag_id": "ready",
                    "platform": "operational",
                    "package_path": "ready.yaml",
                    "modes": [{"name": "kubernetes", "gpu_floor": 0, "pools": []}],
                },
                {
                    "dag_id": "blocked",
                    "platform": "kubernetes",
                    "package_path": "blocked.yaml",
                    "modes": [{"name": "internal", "gpu_floor": 2, "pools": []}],
                },
            ]
            kube = preflight.KubernetesFacts(
                reachable=True,
                supported=True,
                free_gpus=0,
                cleanup_permissions=True,
            )
            controller = preflight.ControllerFacts(
                installed=True,
                healthy=True,
                api_verified=True,
                dags={
                    "ready": {"dag_id": "ready", "is_paused": False},
                    "blocked": {"dag_id": "blocked", "is_paused": False},
                },
            )
            workflows, overall = preflight.workflow_report(repo, profiles, {}, kube, controller)
            self.assertEqual(overall, preflight.PARTIAL)
            self.assertEqual(
                [item["status"] for item in workflows],
                [preflight.READY, preflight.BLOCKED],
            )

    def test_package_rename_does_not_accept_the_old_deployed_dag_id(self) -> None:
        profile = {
            "dag_id": "autonomous_vehicle_dataset_generation_dag_k8s",
            "platform": "operational",
            "package_path": "current.yaml",
            "legacy_package_paths": ["legacy.yaml"],
            "modes": [{"name": "kubernetes", "gpu_floor": 0, "pools": []}],
        }
        kube = preflight.KubernetesFacts(
            reachable=True, supported=True, free_gpus=0, cleanup_permissions=True
        )
        controller = preflight.ControllerFacts(
            installed=True,
            healthy=True,
            api_verified=True,
            dags={"curation_and_retrieval_dag_k8s": {"is_paused": False}},
        )
        for package in ("current.yaml", "legacy.yaml", None):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                if package:
                    (repo / package).write_text("x")
                workflows, _ = preflight.workflow_report(repo, [profile], {}, kube, controller)
                expected = preflight.BLOCKED if package else preflight.NOT_PACKAGED
                self.assertEqual(workflows[0]["status"], expected)

    def test_unprofiled_live_dag_is_never_silently_ready(self) -> None:
        controller = preflight.ControllerFacts(
            installed=True,
            healthy=True,
            api_verified=True,
            dags={"new_dag": {"dag_id": "new_dag", "is_paused": False}},
        )
        workflows, overall = preflight.workflow_report(
            Path("/nonexistent"), [], {}, preflight.KubernetesFacts(), controller
        )
        self.assertEqual(overall, preflight.UNVERIFIED)
        self.assertEqual(workflows[0]["status"], preflight.UNVERIFIED)


if __name__ == "__main__":
    unittest.main()
