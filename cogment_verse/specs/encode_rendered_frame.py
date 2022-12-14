# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import datetime



def encode_rendered_frame(rendered_frame, max_size=1024):
    if max_size <= 0:
        max_size = 1024
    # gRPC max message size hack
    height, width = rendered_frame.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(new_height / height * width)
        else:
            new_width = max_size
            new_height = int(height / width * new_width)
        rendered_frame = cv2.resize(rendered_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # note rgb -> bgr for cv2
    result, encoded_frame = cv2.imencode(".jpg", rendered_frame[:, :, ::-1])
    # cv2.imshow("Image", result)

    # dt_now = datetime.datetime.now()
    # cv2.imwrite(f"testoutput/output{dt_now.microsecond}.jpg", rendered_frame[:, :, ::-1])

    assert result
    # print(os.getcwd())
    # encoded_frame = cv2.imread("./render.png")
    # print(rendered_frame)
    # print(encoded_frame.tobytes())

    return encoded_frame.tobytes()
    # return encoded_frame
