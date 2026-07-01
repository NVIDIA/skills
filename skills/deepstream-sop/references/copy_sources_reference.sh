######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
######################################################################################################

set -e

if [[ "$#" -ne 2 ]]; then
    echo "Error: Exactly two arguments required."
    echo "Usage: $(basename $0) <src-dir> <out-dir>"
    exit 1
fi

VIA_SRC_DIR=$(realpath $1)
OUT_DIR=$(realpath $2)
SCRIPT_DIR=$(dirname $(realpath $0))

FILE_LIST="$SCRIPT_DIR/package_file_list.txt"

while read -r file || [[ -n "$file" ]]; do
    if [[ $file = \#* ]] || [[ -z "$file" ]] ; then
        continue
    fi
    SRC_FILE="$VIA_SRC_DIR/$file"
    DEST_FILE="$OUT_DIR/$file"
    DEST_DIR=$(dirname $DEST_FILE)
    mkdir -p "$DEST_DIR"
    cp -rv "$SRC_FILE" "$DEST_FILE"
done < $FILE_LIST
