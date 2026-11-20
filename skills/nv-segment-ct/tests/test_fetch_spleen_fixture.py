# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import io
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "fetch_spleen_fixture.py"
spec = importlib.util.spec_from_file_location("fetch_spleen_fixture", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


@pytest.mark.parametrize("announced_size", [4, 8])
def test_download_uses_fixed_https_object_and_checks_length(tmp_path, monkeypatch, announced_size):
    def respond(opener, url, *, timeout):
        assert url == module.TASK09_URL and url.startswith("https://")
        assert timeout > 0
        assert not any(hasattr(handler, "redirect_request") for handler in opener.handlers)
        response = io.BytesIO(b"data")
        response.headers = {"Content-Length": str(announced_size)}
        return response

    monkeypatch.setattr(module.urllib.request.OpenerDirector, "open", respond)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    dest = tmp_path / "fixture.tar"
    if announced_size == 4:
        module._download(dest)
        assert dest.read_bytes() == b"data"
    else:
        with pytest.raises(OSError, match="Incomplete fixture"):
            module._download(dest)
        assert not dest.exists()
