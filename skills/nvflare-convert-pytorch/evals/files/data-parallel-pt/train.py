# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from model import DataParallelNetwork


def main():
    device = torch.device("cuda")
    model = torch.nn.DataParallel(DataParallelNetwork()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features = torch.randn(16, 4, device=device)
    labels = torch.randint(0, 2, (16,), device=device)

    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
