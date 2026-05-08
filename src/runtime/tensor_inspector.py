import torch


def tensor_memory_mb(tensor: torch.Tensor) -> float:
    """
    Estimate how much memory a tensor uses in megabytes.

    Formula:
        number of elements × bytes per element
    """

    total_bytes = tensor.numel() * tensor.element_size()
    total_mb = total_bytes / (1024 * 1024)
    return total_mb


def explain_shape(tensor: torch.Tensor) -> str:
    """
    Explain common tensor shapes used in inference systems.
    """

    shape = list(tensor.shape)
    rank = tensor.dim()

    if rank == 0:
        return "Scalar tensor: one single value."

    if rank == 1:
        return f"Vector tensor: likely {shape[0]} features or token IDs."

    if rank == 2:
        return (
            f"2D tensor: often [batch_size, features] or "
            f"[batch_size, sequence_length]. Batch size = {shape[0]}."
        )

    if rank == 3:
        return (
            "3D tensor: could be one image [channels, height, width], "
            "or sequence embeddings [batch, sequence, hidden]."
        )

    if rank == 4:
        return (
            "4D tensor: often image batch [batch, channels, height, width]. "
            f"Batch size = {shape[0]}."
        )

    if rank == 5:
        return (
            "5D tensor: often video batch [batch, frames, channels, height, width]. "
            f"Batch size = {shape[0]}."
        )

    return f"{rank}D tensor: advanced tensor shape."


def inspect_tensor(name: str, tensor: torch.Tensor) -> None:
    """
    Print useful debugging information about a tensor.
    """

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Shape: {list(tensor.shape)}")
    print(f"Rank / dimensions: {tensor.dim()}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Number of elements: {tensor.numel()}")
    print(f"Bytes per element: {tensor.element_size()}")
    print(f"Estimated memory: {tensor_memory_mb(tensor):.6f} MB")
    print(f"Shape explanation: {explain_shape(tensor)}")


def create_example_tensors() -> dict[str, torch.Tensor]:
    """
    Create example tensors that resemble different inference workloads.
    """

    examples = {
        # One input with 4 numeric features.
        "single_feature_batch": torch.randn(1, 4, dtype=torch.float32),

        # Eight inputs, each with 4 features.
        "larger_feature_batch": torch.randn(8, 4, dtype=torch.float32),

        # Text token IDs: batch of 2 text prompts, each 6 tokens long.
        "text_token_ids": torch.tensor(
            [
                [101, 2023, 2003, 1037, 3231, 102],
                [101, 7592, 2088, 0, 0, 0],
            ],
            dtype=torch.int64,
        ),

        # Attention mask: 1 means real token, 0 means padding.
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.int64,
        ),

        # Batch of 4 RGB images, each 224x224.
        "image_batch": torch.randn(4, 3, 224, 224, dtype=torch.float32),

        # Batch of 2 audio waveforms, each with 16000 samples.
        "audio_batch": torch.randn(2, 16000, dtype=torch.float32),

        # Batch of 1 video, with 8 frames, RGB, 224x224.
        "video_batch": torch.randn(1, 8, 3, 224, 224, dtype=torch.float32),

        # Classifier output logits: batch of 4 examples, 10 classes.
        "classifier_logits": torch.randn(4, 10, dtype=torch.float32),
    }

    return examples


def demonstrate_dtype_memory() -> None:
    """
    Show how dtype affects tensor memory.
    """

    x_float32 = torch.randn(8, 3, 224, 224, dtype=torch.float32)
    x_float16 = x_float32.to(torch.float16)
    x_bfloat16 = x_float32.to(torch.bfloat16)

    inspect_tensor("image_batch_float32", x_float32)
    inspect_tensor("image_batch_float16", x_float16)
    inspect_tensor("image_batch_bfloat16", x_bfloat16)


def demonstrate_device_movement() -> None:
    """
    Show how to move tensors from CPU to GPU when available.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_cpu = torch.randn(2, 4)
    x_device = x_cpu.to(device)

    inspect_tensor("x_cpu", x_cpu)
    inspect_tensor(f"x_on_{device}", x_device)


def main() -> None:
    print("OmniServe Day 2 — Tensor Inspector")

    examples = create_example_tensors()

    for name, tensor in examples.items():
        inspect_tensor(name, tensor)

    print("\n" + "=" * 80)
    print("Dtype memory demonstration")
    print("=" * 80)
    demonstrate_dtype_memory()

    print("\n" + "=" * 80)
    print("Device movement demonstration")
    print("=" * 80)
    demonstrate_device_movement()


if __name__ == "__main__":
    main()