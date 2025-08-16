import torch
import onnx, onnxscript
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase


@torch.inference_mode()
def export_onnx(module: ModBase, td: TensorDictBase, path: str, meta=None):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    
    onnx_program = torch.onnx.export(
        module,
        kwargs=td.to_dict(),
        dynamo=True,
    )
    onnx_program.save(path)
    print(f"Exported ONNX model to {path}.")

    import json

    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["torch_version"] = str(torch.__version__)
    meta["onnx_version"] = str(onnx.__version__)
    meta["onnxscript_version"] = str(onnxscript.__version__)
    meta["in_keys"] = module.in_keys
    meta["out_keys"] = module.out_keys
    meta["in_shapes"] = ([td[k].shape for k in module.in_keys],)

    json.dump(meta, open(meta_path, "w"), indent=4)
    print(f"Exported metadata to {meta_path}.")
    print(f"torch version: {torch.__version__}, onnx version: {onnx.__version__}, onnxscript version: {onnxscript.__version__}")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        path.replace(".pt", ".onnx"), providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnx_input = tuple(td[k] for k in module.in_keys)
    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_output = ort_session.run(None, onnxruntime_input)
    assert len(ort_output) == len(module.out_keys)
