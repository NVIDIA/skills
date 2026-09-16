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


def build_optimizer(parameters):
    return torch.optim.SGD(parameters, lr=0.1)


model = torch.nn.Linear(2, 2)
features = torch.randn(8, 2)
labels = torch.randint(0, 2, (8,))
optimizer = build_optimizer(model.parameters())

for _ in range(2):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
