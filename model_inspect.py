#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
"""

import onnx
from typing import List


def _format_shape(vi) -> str:
    """Return a human-readable shape string from a ValueInfoProto."""
    dims: List[str] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(str(d.dim_value))
    return "(" + ", ".join(dims) + ")"


def inspect_onnx(model_path: str) -> None:
    print(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    print("\n=== MODEL INPUTS ===")
    for i, inp in enumerate(graph.input):
        print(f"[{i}] name={inp.name}, shape={_format_shape(inp)}")

    print("\n=== MODEL OUTPUTS ===")
    for i, out in enumerate(graph.output):
        print(f"[{i}] name={out.name}, shape={_format_shape(out)}")

    # Optional: dump a short printable graph for debugging
    print("\n=== SHORT GRAPH DUMP (first 100 lines) ===")
    txt = onnx.helper.printable_graph(graph)
    lines = txt.splitlines()
    for line in lines[:100]:
        print(line)
    if len(lines) > 100:
        print("... (truncated) ...")


if __name__ == "__main__":
    # Adjust path if needed
    inspect_onnx("yolov8x-p2_opt_4o.onnx")
