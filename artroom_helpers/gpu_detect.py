import torch


# Need to expand to include more options, 10XX, 20XX, 30XX for auto xformers
def get_gpu_architecture():
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        if '1630' in gpu_info or '1650' in gpu_info or '1660' in gpu_info or '1600' in gpu_info:
            print(gpu_info + ' identified, forcing to full precision')
            return '16XX'
        return 'NVIDIA'
    else:
        try:
            import torch_directml
            print("Directml supported")
            return "DIRECTML"
        except:
            print("Cuda not available.")
            print("If you are using NVIDIA GPU please try updating your drivers")
            print("If you are using AMD, it is not yet supported.")
            return 'None'


def get_device():
    try:
        import torch_directml
    except:
        pass
    match get_gpu_architecture():
        case 'NVIDIA' | '16XX':
            return torch.device(0)
        case "DIRECTML":
            return torch_directml.device()
        case _:
            return torch.device("cpu")
