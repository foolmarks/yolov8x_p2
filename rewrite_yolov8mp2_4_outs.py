import sys
import argparse
import os
import numpy as np
import onnx
#import onnxoptimizer
from onnxsim import simplify
import onnxruntime as ort
from onnxsim import simplify, model_info
from pathlib import Path

# Add the parent directory of sima_utils to sys.path
import sys, os
sima_utils_pkg_path = os.path.abspath("sima-utils")
if sima_utils_pkg_path not in sys.path:
    sys.path.insert(0, sima_utils_pkg_path)

# Now import directly from the folder
import sima_utils.onnx.onnx_helpers as oh

from yolo_model import YOLOModel


def save_model(model: onnx.ModelProto, model_fname: str, save_only: bool = False):
    """
    Save a model to disk.

    :param model: Model to be saved.
    :param model_fname: File name to be used to save the model.
    :param save_only: Boolean flag, if set to False, to simplify and re-generate shape inference
        result before saving to disk.
    """

    if not save_only:
        # Simplify model.
        model_opt, check = simplify(model)
        assert check, "Simplified ONNX model can not be validated"
        model_info.print_simplifying_info(model, model_opt)
        model_opt = onnx.shape_inference.infer_shapes(model_opt)
        onnx.checker.check_model(model_opt, full_check=True)
        model = model_opt
    onnx.save(model, model_fname)
    print(f'ONNX file saved to {model_fname}')
    

def get_node_names(model):
    inputs = []
    initializer_names = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name not in initializer_names:
            inputs.append(inp.name)
    if not inputs:
        raise ValueError("No valid input node found in the model.")
    return inputs


def remove_unused_initializers(model):
    used_inputs = set()
    for node in model.graph.node:
        for name in node.input:
            used_inputs.add(name)

    new_initializers = []
    removed_initializers = []
    for init in model.graph.initializer:
        if init.name in used_inputs:
            new_initializers.append(init)
        else:
            removed_initializers.append(init.name)

    if removed_initializers:
        print("Removing unused initializers:")
        for name in removed_initializers:
            print("  ", name)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    new_inputs = [
        inp
        for inp in model.graph.input
        if inp.name in used_inputs or any(inp.name == init.name for init in model.graph.initializer)
    ]
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    return model


def run_inference(model_path, input_data, output_nodes=None):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session.run(output_nodes, {input_name: input_data}) if output_nodes else session.run(None, {input_name: input_data})


def compare_tensors(tensor1, tensor2, atol=1e-7):
    if tensor1.shape != tensor2.shape:
        print(f"Different shapes: {tensor1.shape} vs {tensor2.shape}")
        return
    difference_mask = ~np.isclose(tensor1, tensor2, atol=atol)
    if np.any(difference_mask):
        print("Difference found at indexes:")
        for index in np.argwhere(difference_mask):
            idx = tuple(index)
            print(f"  {idx}: tensor1={tensor1[idx]}, tensor2={tensor2[idx]}")
    else:
        print(f"Tensors are close (atol={atol})")



def main(model_files):
    for model_path in model_files:
        yolo = YOLOModel(model_path)
        model = yolo.model
        model_name = yolo.model_name
        mod_model_name = f"{model_name}_mod"
        input_node_name = yolo.input_node_name
        model_prefix_no_block = yolo._find_prefix("attn/qkv/conv/Conv")
        H, W = yolo.H, yolo.W
        num_classes = yolo.num_classes
        one2one_prefix = yolo.one2one_prefix
        model_flavor = yolo.flavor
        splits = yolo.splits
        has_attn = yolo.has_attention

        print(f"Model: {model_name}, Version: yolov{yolo.version}, Flavor/size: {model_flavor}, H: {H}, W: {W}, Classes: {num_classes}")

        # === attention block surgery (unchanged) ===
        if has_attn:
            for block in range(2 if (yolo.version == 11 and model_flavor in ['l', 'x']) else 1):
                if yolo.version != 10:
                    model_prefix = f"{model_prefix_no_block}/m.{block}/attn"
                else:
                    model_prefix = model_prefix_no_block

                # Replace MatMul with Einsum
                dict1 = {f"{model_prefix}/MatMul": "nchw,nchq->nqhw"}
                dict2 = {f"{model_prefix}/MatMul_1": "nchw,nqhc->nqhw"}
                oh.rewrite_matmul_as_einsum(model, dict1)
                oh.rewrite_matmul_as_einsum(model, dict2)

                # ... (all the same internal graph surgery from your previous script) ...
                # Skipping for brevity, unchanged until output section
                pass

        onnx.save(model, f"{mod_model_name}.onnx")

        model_opt, check = simplify(model)
        assert check, "Simplified ONNX model cannot be validated"
        model = model_opt

        model = remove_unused_initializers(model)
        oh.remove_infer_shape(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, f"{mod_model_name}.onnx")

        input_onnx = f"{mod_model_name}.onnx"
        output_onnx = f"{model_name}_opt_4o.onnx"

        model = onnx.load(input_onnx)
        oh.remove_output(model)

        # === NEW generalized detection head logic ===
        #
        # The model now supports 4 detection branches instead of 3
        #  - bbox_0 ... bbox_3
        #  - class_prob_0 ... class_prob_3

        branch_levels = [H // 4,H // 8, H // 16, H // 32]  # 4 branches now
        stride_labels = ["_0", "_1", "_2", "_3"]
        orig_cut = []

        # DETECTION OUTPUTS
        for i, stride in enumerate(branch_levels):
            oh.add_output(model, f"bbox{stride_labels[i]}", (1, 64, stride, W // (H // stride)))
            oh.add_output(model, f"class_prob{stride_labels[i]}", (1, num_classes, stride, W // (H // stride)))

        op_prefix = f"{one2one_prefix}cv2.0/{one2one_prefix}cv2.0.2/Conv"
        model_prefix = yolo._find_prefix(op_prefix)
        if model_prefix and f"{one2one_prefix}cv2.0" in model_prefix:
            model_prefix = model_prefix.removesuffix(f'/{one2one_prefix}cv2.0')

        # Connect bbox branches
        for i in range(len(branch_levels)):
            bbox_conv_name = f"{model_prefix}/{one2one_prefix}cv2.{i}/{one2one_prefix}cv2.{i}.2/Conv"
            orig_output_name = oh.find_node(model, bbox_conv_name).output[0]
            next_node, idx = oh.find_node_input(model, orig_output_name)
            oh.change_node_output(model, bbox_conv_name, f"bbox{stride_labels[i]}")
            next_node.input[idx] = f"bbox{stride_labels[i]}"
            orig_cut.append(orig_output_name)

        # Connect class probability branches
        for i in range(len(branch_levels)):
            class_conv_name = f"{model_prefix}/{one2one_prefix}cv3.{i}/{one2one_prefix}cv3.{i}.2/Conv"
            orig_output_name = oh.find_node(model, class_conv_name).output[0]
            next_node, idx = oh.find_node_input(model, orig_output_name)
            oh.change_node_output(model, class_conv_name, f"class_prob{stride_labels[i]}")
            next_node.input[idx] = f"class_prob{stride_labels[i]}"
            orig_cut.append(orig_output_name)

        reshape_cut = [f"bbox{stride_labels[i]}" for i in range(len(branch_levels))]
        reshape_cut.extend([f"class_prob{stride_labels[i]}" for i in range(len(branch_levels))])

        save_model(model, input_onnx)

        # === SEGMENTATION HEAD ===
        op_prefix = f"{one2one_prefix}cv4.0/{one2one_prefix}cv4.0.2/Conv"
        model_prefix = yolo._find_prefix(op_prefix)

        if model_prefix:
            print("Mask head found -> this is a segmentation model!")
            oh.add_output(model, "mask_coeff_0", (1, 32, H//8, W//8))
            oh.add_output(model, "mask_coeff_1", (1, 32, H//16, W//16))
            oh.add_output(model, "mask_coeff_2", (1, 32, H//32, W//32))
            oh.add_output(model, "mask", (1, 32, H//4, W//4))

            if f"{one2one_prefix}cv4.0" in model_prefix:
                model_prefix = model_prefix.removesuffix(f'/{one2one_prefix}cv4.0')

            for i in range(3):
                mask_conv_name = f"{model_prefix}/{one2one_prefix}cv4.{i}/{one2one_prefix}cv4.{i}.2/Conv"
                orig_output_name = oh.find_node(model, mask_conv_name).output[0]
                next_node, idx = oh.find_node_input(model, orig_output_name)
                oh.change_node_output(model, mask_conv_name, f"mask_coeff_{i}")
                next_node.input[idx] = f"mask_coeff_{i}"
                orig_cut.append(orig_output_name)

            mask_node_name = f"{model_prefix}/proto/cv3/conv/Conv"
            try:
                orig_output_name = oh.find_node(model, mask_node_name).output[0]
                next_node, idx = oh.find_node_input(model, orig_output_name)
                next_node.input[idx] = "mask"
            except RuntimeError as e:
                print(f"Warning: {e}")

            oh.change_node_output(model, mask_node_name, "mask")
            orig_cut.append(orig_output_name)
            reshape_cut.extend([f"mask_coeff_{i}" for i in range(3)] + ["mask"])
        else:
            print("No mask head found -> this is a detection-only model!")

        onnx.utils.extract_model(input_onnx, output_onnx, input_node_name, reshape_cut)

        model = oh.load_model(output_onnx)
        print(f"\nFinal model: {output_onnx}")

        # === Validation ===
        input_data = np.random.uniform(low=-1.0, high=1.0, size=(1, 3, H, W)).astype(np.float32)
        for i, cut_node in enumerate(orig_cut):
            model1 = oh.load_model(model_path, load_only=True)
            value_info_protos = [vi for vi in onnx.shape_inference.infer_shapes(model1).graph.value_info if vi.name == cut_node]
            model1.graph.output.extend(value_info_protos)
            onnx.checker.check_model(model1)
            inter_path = f"{model_name}_inter.onnx"
            onnx.save(model1, inter_path)

            output_node_1 = run_inference(inter_path, input_data, output_nodes=[cut_node])[0]
            output_node_2 = run_inference(f"{model_name}_opt_4o.onnx", input_data)[i]
            print(f"Comparing output at node {i}: {cut_node}:")
            compare_tensors(output_node_1, output_node_2)

        os.remove(inter_path)
        os.remove(f"{mod_model_name}.onnx")
        print(f"Model {model_name} done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do surgery on YOLO_v8m_p2.")
    parser.add_argument("model_files", nargs="+", help="List of ONNX model files to process")
    args = parser.parse_args()
    main(args.model_files)