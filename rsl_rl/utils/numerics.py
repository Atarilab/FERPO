from __future__ import annotations

import torch


def tensor_issue_description(
    name: str,
    tensor: torch.Tensor,
    invalid: torch.Tensor,
    *,
    environment_axis: int | None = None,
    max_examples: int = 8,
) -> str | None:
    """Describe invalid tensor elements while keeping healthy-path work minimal."""
    if not bool(invalid.any().item()):
        return None
    coordinates = torch.nonzero(invalid, as_tuple=False)
    example_coordinates = coordinates[:max_examples]
    example_values = tensor[invalid][:max_examples]
    details = [
        f"{name} contains {int(invalid.sum().item())}/{tensor.numel()} invalid values",
        f"sample_indices={example_coordinates.detach().cpu().tolist()}",
        f"sample_values={example_values.detach().cpu().tolist()}",
    ]
    if environment_axis is not None and coordinates.shape[1] > environment_axis:
        environments = coordinates[:, environment_axis].unique()[:max_examples]
        details.append(f"environment_indices={environments.detach().cpu().tolist()}")
    return "; ".join(details)


def non_finite_tensor_description(
    name: str,
    tensor: torch.Tensor,
    *,
    environment_axis: int | None = None,
    max_examples: int = 8,
) -> str | None:
    return tensor_issue_description(
        name,
        tensor,
        ~torch.isfinite(tensor),
        environment_axis=environment_axis,
        max_examples=max_examples,
    )


def finite_tensor_range(name: str, tensor: torch.Tensor) -> str:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return f"{name}=no finite values"
    return f"{name}_range=[{float(finite.min().item()):.6g}, {float(finite.max().item()):.6g}]"
