# Copyright 2018 Jaewook Kang and JoonHo Lee ({jwkang10, junhoning}@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-
import sys


def progress_bar(total, progress, state_msg):
    """
    Displays or updates a console progress bar.
    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"

    block = int(round(barLength * progress))
    progress_bar = "\r [{}] {:.0f}% -> {}{}".format("#" * block + "-" * (barLength - block),
                                                    round(progress * 100, 0),
                                                    state_msg,
                                                    status)
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
