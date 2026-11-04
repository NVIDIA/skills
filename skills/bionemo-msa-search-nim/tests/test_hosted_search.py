"""Offline transport and artifact checks; no hosted-service calls are made."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "hosted_search", Path(__file__).resolve().parents[1] / "scripts" / "hosted_search.py"
)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


def response(status=200, data=None):
    result = Mock(status_code=status)
    result.json.return_value = data
    return result


def alignments():
    return {"alignments": {
        db: {"a3m": {"alignment": ">query\nACDE\n>hit\nAC-E\n", "format": "a3m"}}
        for db in client.DATABASES
    }}


class HostedSearchTests(unittest.TestCase):
    def test_gateway_timeout_retries_once_then_stops(self):
        failures = [response(504), response(504), response(data=alignments())]
        with patch.object(client.requests, "post", side_effect=failures) as post, \
                patch.object(client.time, "sleep"):
            with self.assertRaisesRegex(client.SearchError, "HTTP 504.*Stopped after 2"):
                client.search("ACDE", list(client.DATABASES), "test-credential")
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["timeout"], (10, 300))
            self.assertFalse(call.kwargs["allow_redirects"])

    def test_transient_error_can_recover_without_changing_the_request(self):
        expected = alignments()
        with patch.object(client.requests, "post", side_effect=[response(503), response(data=expected)]) as post, \
                patch.object(client.time, "sleep"):
            actual = client.search("ACDE", list(client.DATABASES), "test-credential")
        self.assertEqual(actual, expected)
        self.assertEqual(post.call_args_list[0], post.call_args_list[1])

    def test_auth_error_and_redirect_are_not_retried(self):
        for status in [401, 403, 302]:
            with self.subTest(status=status), patch.object(client.requests, "post", return_value=response(status)) as post:
                with self.assertRaisesRegex(client.SearchError, f"HTTP {status}"):
                    client.search("ACDE", list(client.DATABASES), "test-credential")
                self.assertEqual(post.call_count, 1)

    def test_timeout_does_not_expose_exception_text(self):
        with patch.object(client.requests, "post", side_effect=client.requests.ReadTimeout("sensitive request details")) as post, \
                patch.object(client.time, "sleep"):
            with self.assertRaises(client.SearchError) as failure:
                client.search("ACDE", list(client.DATABASES), "test-credential")
        self.assertNotIn("sensitive", str(failure.exception))
        self.assertEqual(post.call_count, 2)

    def test_partial_or_empty_alignment_is_not_success(self):
        partial = alignments()
        del partial["alignments"][client.DATABASES[1]]
        empty = alignments()
        empty["alignments"][client.DATABASES[0]]["a3m"]["alignment"] = ">query\n"
        for result in [None, {}, partial, empty]:
            with self.subTest(result=result), patch.object(client.requests, "post", return_value=response(data=result)):
                with self.assertRaises(client.SearchError):
                    client.search("ACDE", list(client.DATABASES), "test-credential")

    def test_alignment_format_must_be_a3m(self):
        for returned_format in [None, "fasta", "A3M", 1]:
            result = alignments()
            a3m = result["alignments"][client.DATABASES[0]]["a3m"]
            if returned_format is None:
                del a3m["format"]
            else:
                a3m["format"] = returned_format
            with self.subTest(format=returned_format), \
                    patch.object(client.requests, "post", return_value=response(data=result)) as post:
                with self.assertRaises(client.SearchError):
                    client.search("ACDE", list(client.DATABASES), "test-credential")
                self.assertEqual(post.call_count, 1)

    def test_malformed_a3m_is_not_success(self):
        malformed = [
            None,
            "",
            "ACDE\n",
            "ACDE\n>query\nACDE\n",
            ">query\n>hit\nACDE\n",
            ">query\nACDE\n>hit\n",
            ">query\nACDE\n>empty\n>hit\nACDE\n",
            "> \nACDE\n",
            " >query\nACDE\n",
            ">query\nAC DE\n",
            ">query\nACD1\n",
            ">query\nACD*\n",
            ">query\nACDÉ\n",
            ">query\nACDE\n>hit\nACD\n",
            ">query\nacde\n",
            ">query\n# no sequence\n",
        ]
        for alignment in malformed:
            result = alignments()
            result["alignments"][client.DATABASES[1]]["a3m"]["alignment"] = alignment
            reply = response(data=result)
            with self.subTest(alignment=alignment), \
                    patch.object(client.requests, "post", return_value=reply) as post:
                with self.assertRaises(client.SearchError):
                    client.search("ACDE", list(client.DATABASES), "test-credential")
                self.assertEqual(post.call_count, 1)
                reply.close.assert_called_once()

    def test_valid_a3m_supports_wrapping_insertions_and_comments(self):
        for alignment in [
            ">query\nACDE",
            ">query description\nAC\nDE\n>hit\nAcC-\nE\n",
            "# comment\n\n>query\nACDE\n\n>hit\nacACd-Efg\n# comment\n",
        ]:
            expected = alignments()
            for database in client.DATABASES:
                expected["alignments"][database]["a3m"]["alignment"] = alignment
            with self.subTest(alignment=alignment), \
                    patch.object(client.requests, "post", return_value=response(data=expected)):
                self.assertEqual(client.search("ACDE", list(client.DATABASES), "test-credential"), expected)

    def test_missing_key_and_unknown_database_fail_before_network(self):
        for databases, key in [(list(client.DATABASES), ""), (["../result"], "test-credential")]:
            with self.subTest(databases=databases), patch.object(client.requests, "post") as post:
                with self.assertRaises(client.SearchError):
                    client.search("ACDE", databases, key)
                post.assert_not_called()

    def test_cli_writes_actual_response_and_alignments(self):
        expected = alignments()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "msa"
            args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]
            with patch.object(client.sys, "argv", args), patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                    patch.object(client.requests, "post", return_value=response(data=expected)), redirect_stdout(io.StringIO()):
                self.assertEqual(client.main(), 0)
            self.assertEqual(json.loads((output / "response.json").read_text()), expected)
            for database in client.DATABASES:
                self.assertEqual((output / f"{database}.a3m").read_text(), expected["alignments"][database]["a3m"]["alignment"])

    @unittest.skipUnless(client.os.name == "posix", "Requires POSIX permissions")
    def test_cli_output_is_private_with_permissive_umask(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root).chmod(0o755)
            output = Path(root) / "msa"
            args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]
            previous_umask = client.os.umask(0)
            try:
                with patch.object(client.sys, "argv", args), \
                        patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                        patch.object(client.requests, "post", return_value=response(data=alignments())), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(client.main(), 0)
            finally:
                client.os.umask(previous_umask)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            for path in output.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path.name)

    def test_cli_does_not_replace_concurrent_empty_output_directory(self):
        mkdir, rename = Path.mkdir, Path.rename
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "msa"
            args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]
            competing_stat = None

            def reserve_output():
                nonlocal competing_stat
                mkdir(output, mode=0o700)
                competing_stat = output.stat()

            # Simulate a competing reservation immediately before publication,
            # after any existence check, with either directory operation.
            def racing_mkdir(path, *args, **kwargs):
                if path == output:
                    reserve_output()
                return mkdir(path, *args, **kwargs)

            def racing_rename(path, target):
                if Path(target) == output:
                    reserve_output()
                return rename(path, target)

            with patch.object(client.sys, "argv", args), \
                    patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                    patch.object(client.requests, "post", return_value=response(data=alignments())), \
                    patch.object(Path, "mkdir", racing_mkdir), \
                    patch.object(Path, "rename", racing_rename), \
                    redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                self.assertEqual(client.main(), 1)
            self.assertIsNotNone(competing_stat)
            self.assertEqual(output.stat().st_ino, competing_stat.st_ino)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(Path(root).iterdir()), [output])
            self.assertEqual(stdout.getvalue(), "")

    def test_cli_failure_leaves_no_success_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "msa"
            args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]
            with patch.object(client.sys, "argv", args), patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                    patch.object(client.requests, "post", return_value=response(504)), \
                    patch.object(client.time, "sleep"), redirect_stderr(io.StringIO()):
                self.assertEqual(client.main(), 1)
            self.assertFalse(output.exists())

    def test_cli_write_failure_cleans_up_and_allows_retry(self):
        write_text = Path.write_text
        filenames = ["response.json", *(f"{database}.a3m" for database in client.DATABASES)]
        for failing_file in filenames:
            with self.subTest(file=failing_file), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "msa"
                expected = alignments()
                args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]

                def fail_during_write(path, data, **kwargs):
                    if path.name == failing_file:
                        write_text(path, "partial", **kwargs)
                        raise OSError("sensitive filesystem details")
                    return write_text(path, data, **kwargs)

                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(client.sys, "argv", args), \
                        patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                        patch.object(client.requests, "post", return_value=response(data=expected)):
                    with patch.object(Path, "write_text", fail_during_write), \
                            redirect_stdout(stdout), redirect_stderr(stderr):
                        self.assertEqual(client.main(), 1)
                    self.assertEqual(list(Path(root).iterdir()), [])
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertNotIn("sensitive", stderr.getvalue())
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(client.main(), 0)
                self.assertEqual(json.loads((output / "response.json").read_text()), expected)
                for database in client.DATABASES:
                    self.assertEqual((output / f"{database}.a3m").read_text(),
                                     expected["alignments"][database]["a3m"]["alignment"])

    def test_cli_publish_failure_leaves_no_artifacts(self):
        rename = Path.rename
        filenames = ["response.json", *(f"{database}.a3m" for database in client.DATABASES)]
        for failing_file in filenames:
            with self.subTest(file=failing_file), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "msa"
                args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]

                def fail_during_publish(path, target):
                    if path.name == failing_file:
                        raise OSError("cannot publish")
                    return rename(path, target)

                with patch.object(client.sys, "argv", args), \
                        patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                        patch.object(client.requests, "post", return_value=response(data=alignments())):
                    with patch.object(Path, "rename", fail_during_publish), \
                            redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                        self.assertEqual(client.main(), 1)
                    self.assertEqual(list(Path(root).iterdir()), [])
                    self.assertEqual(stdout.getvalue(), "")
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(client.main(), 0)
                self.assertEqual({path.name for path in output.iterdir()}, set(filenames))

    def test_cli_does_not_overwrite_output_created_during_search(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "msa"
            args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]

            def concurrent_output(*args, **kwargs):
                output.mkdir()
                (output / "keep.txt").write_text("existing data")
                return response(data=alignments())

            with patch.object(client.sys, "argv", args), \
                    patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                    patch.object(client.requests, "post", side_effect=concurrent_output), \
                    redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                self.assertEqual(client.main(), 1)
            self.assertEqual(list(Path(root).iterdir()), [output])
            self.assertEqual(list(output.iterdir()), [output / "keep.txt"])
            self.assertEqual((output / "keep.txt").read_text(), "existing data")
            self.assertEqual(stdout.getvalue(), "")

    def test_cli_malformed_response_leaves_no_output(self):
        malformed = alignments()
        malformed["alignments"][client.DATABASES[1]]["a3m"]["alignment"] = ">query\nACDE\n>hit\n"
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "msa"
            args = ["hosted_search.py", "--sequence", "ACDE", "--output-dir", str(output)]
            with patch.object(client.sys, "argv", args), \
                    patch.dict(client.os.environ, {"NGC_API_KEY": "test-credential"}), \
                    patch.object(client.requests, "post", return_value=response(data=malformed)), \
                    redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                self.assertEqual(client.main(), 1)
            self.assertEqual(list(Path(root).iterdir()), [])
            self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
