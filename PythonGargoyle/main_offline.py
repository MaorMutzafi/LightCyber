import torch
from Algorithm.AlgoRunner import AlgoRunner
from Algorithm.CoreFuncs import load_data, unbit_pack_data_torch, LogPoint
from colorama import Fore

def main():
    torch.set_grad_enabled(False)
    
    H = 268
    W = 640
    # fp = 'C:/Users/User/Documents/Mafaat_new_Topics/LightCyber/Exp/Kfir_11June2023/Data/OS_12062023024132_70m_4mW/Vid.raw'
    # offset_path = 'C:/Users/User/Documents/Mafaat_new_Topics/LightCyber/Exp/Kfir_11June2023/Data/GainOffset/Offset.raw'
    # gain_path = 'C:/Users/User/Documents/Mafaat_new_Topics/LightCyber/Exp/Kfir_11June2023/Data/GainOffset/Gain.raw'
    
    fp =          "C:/Users/User/Dropbox/WorkResearch/MafaatProjects/Projects/LightCyber/LightCyberApp/Data/Vid_tst_25Feb24.raw"
    offset_path = "C:/Users/User/Dropbox/WorkResearch/MafaatProjects/Projects/LightCyber/LightCyberApp/Data/Offset.txt"
    gain_path =   "C:/Users/User/Dropbox/WorkResearch/MafaatProjects/Projects/LightCyber/LightCyberApp/Data/Gain.txt"
    # fp = "/home/rakia/GargoyleEfcom/data/Vid.raw"
    # offset_path = "/home/rakia/GargoyleEfcom/data/Offset.txt"
    # gain_path = "/home/rakia/GargoyleEfcom/data/Gain.txt"

    algorunner = AlgoRunner(H, W, offset_path=offset_path, gain_path=gain_path)
    print(Fore.MAGENTA + 'Loadin the data from file to numpy')
    
    frames_bp = load_data(fp, H, W)
    
    print(Fore.GREEN + 'Unbit pack and moving to Torch')
    
    mover = LogPoint("Move to cuda")
    frames = unbit_pack_data_torch(frames_bp, H, W)[200:300]    
    mover.end_and_print()
    
    print(Fore.BLUE + 'Frames shape = ' + str(frames.shape))
    print(Fore.BLUE + 'Frames type = ' + str(frames.dtype))
    print(Fore.BLUE + 'Frames device = ' + str(frames.device))
    
    print(Fore.CYAN + 'Run the algorithm')
    
    algo=LogPoint("Algo")
    results = algorunner.accelrated_vid(frames)
    algo.end_and_print()
    
    print(Fore.CYAN + 'The results' + Fore.RESET)
    print(results[0])
    

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    

if __name__ == '__main__':
    main()
