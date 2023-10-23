
def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/fastchat-t5-3b-v1.0",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--gptq", action="store_true", help="Use gptq quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--offload-folder",
        type=str, default=None,
        help="If the device_map contains any value \"disk\", the folder where we will offload weights",
    )
    parser.add_argument(
        "--group-size",
        type=str, default=None,
        help="The group size parameter of quantization",
    )
    parser.add_argument(
        "--deepspeed", action="store_true", help="Use deepspeed"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    parser.add_argument(
        "--sleep", type=int, default=0, help="Sleep after seconds"
    )