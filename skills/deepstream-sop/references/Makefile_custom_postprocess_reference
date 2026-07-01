#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

CC:= g++

CFLAGS:= -Wall -std=c++14 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I../../includes \
         -I /usr/local/cuda/include \
         -I../../includes/nvdsinferserver \
         -I/opt/nvidia/deepstream/deepstream/sources/includes \
         -I/opt/nvidia/deepstream/deepstream/sources/includes/nvdsinferserver

LIBS:= \
       -L/opt/nvidia/deepstream/deepstream/lib/ -lnvds_meta
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvds_custom_postprocess_tensor.cpp

TARGET_LIB:= libnvds_custom_postprocess_tensor.so

PKGS:= glib-2.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
