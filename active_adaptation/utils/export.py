import torch
import json
import yaml
import onnx, onnxscript, onnxruntime as ort
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase

TORCH_VERSION = torch.__version__
ONNX_VERSION = onnx.__version__
ONNXSCRIPT_VERSION = onnxscript.__version__


def to_numpy(tensor: torch.Tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


@torch.inference_mode()
def export_onnx(
    module: ModBase,
    td: TensorDictBase,
    path: str,
    meta=None,
    json_meta: bool = False, # whether to export metadata to json or yaml file
):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")
    print(f"torch version: {TORCH_VERSION}, onnx version: {ONNX_VERSION}, onnxscript version: {ONNXSCRIPT_VERSION}")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    
    input_names = [k if isinstance(k, str) else "_".join(k) for k in module.in_keys]
    output_names = [k if isinstance(k, str) else "_".join(k) for k in module.out_keys]
    onnx_program = torch.onnx.export(
        module,
        kwargs=td.to_dict(),
        dynamo=True,
        verify=True,
        input_names=input_names,
        output_names=output_names,
    )
    onnx_program.save(path)
    print(f"Exported ONNX model to {path}.")

    if meta is None:
        meta = {}
    meta["torch_version"] = str(TORCH_VERSION)
    meta["onnx_version"] = str(ONNX_VERSION)
    meta["onnxscript_version"] = str(ONNXSCRIPT_VERSION)
    meta["in_keys"] = input_names
    meta["out_keys"] = output_names
    meta["in_shapes"] = [list(td[k].shape) for k in module.in_keys]

    if json_meta:
        meta_path = path.replace(".onnx", ".json")
        json.dump(meta, open(meta_path, "w"), indent=4)
    else:
        meta_path = path.replace(".onnx", ".yaml")
        yaml.dump(meta, open(meta_path, "w"), indent=4, default_flow_style=None)

    print(f"Exported metadata to {meta_path}.")

    ort_session = ort.InferenceSession(
        path.replace(".pt", ".onnx"),
        providers=["CPUExecutionProvider"]
    )
    
    onnx_input = {}
    ort_inputs = ort_session.get_inputs()
    if len(ort_inputs) != len(module.in_keys):
        raise RuntimeError(
            f"ONNX input count mismatch: ort={len(ort_inputs)} vs module.in_keys={len(module.in_keys)}"
        )
    # Map ORT inputs by position to preserve the original semantic order even when
    # exporter rewrites names (e.g. nested keys can become next_*_orig).
    for i, input_arg in enumerate(ort_inputs):
        onnx_input[input_arg.name] = to_numpy(td[module.in_keys[i]])
    ort_output = ort_session.run(None, onnx_input)
    assert len(ort_output) == len(module.out_keys)
