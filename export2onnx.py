'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''


'''
Convert PyTorch model to ONNX
'''


from ultralytics import YOLO
from pathlib import Path

model = YOLO("./model.pt")

export_path='model.onnx'


# choose the opset version compatible with your version of Palette SDK
model.export(format="onnx", 
            opset=17,
            simplify=True,
            dynamic=False,
            imgsz=640)


print(f"Exported ONNX model to: {export_path}")

# Rename/move the ONNX file to 'yolov8x-p2.onnx' in the same directory
export_path = Path(export_path)
new_path = export_path.with_name("yolov8x-p2.onnx")

# If a file with that name already exists, delete it
if new_path.exists():
    new_path.unlink()  # delete existing file

export_path.rename(new_path)

print(f"Renamed ONNX model to: {new_path}")

