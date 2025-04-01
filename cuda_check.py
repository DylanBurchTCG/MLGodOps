import torch


def check_cuda():
    """
    Check CUDA availability and print diagnostic information
    """
    print("\n==== CUDA DIAGNOSTICS ====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("\nCUDA is NOT available. Possible reasons:")
        print("1. You don't have an NVIDIA GPU")
        print("2. NVIDIA drivers aren't installed properly")
        print("3. PyTorch was installed without CUDA support")
        print("\nTo install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("(Replace cu118 with your CUDA version)")

    print("==========================\n")
    return torch.cuda.is_available()


if __name__ == "__main__":
    check_cuda()