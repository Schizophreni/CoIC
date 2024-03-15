import torch


def load_ckp(ckp_path):
    return torch.load(ckp_path, map_location="cpu")

def diff_ckp(ckp1, ckp2):
    res = []
    for name, param in ckp1.items():
        if param.dtype == torch.long:
            continue
        diff = (param - ckp2[name]).abs().mean() / param.abs().mean()
        res.append('{}, {:.6f}, {:.6f}, {:.6f}'.format(name, float(diff), float(param.max()), float(ckp2[name].max())))
    return res


if __name__ == "__main__":
    # n63 = "logs/DGUNet-H8L1214-CoIC_DGUNet_128d_0.2contra/net_epoch_136.pth"
    # n65 = "logs/DGUNet-H8L1214-CoIC_DGUNet_128d_0.2contra/net_epoch_200.pth"
    n63 = "logs/DGUNet-H8L1214-CoIC-rerun_DGUNet_128d_0.2contra/net_epoch_35.pth"
    n65 = "logs/DGUNet-H8L1214-CoIC-rerun_DGUNet_128d_0.2contra/net_epoch_40.pth"

    n63, n65 = load_ckp(n63), load_ckp(n65)
    diff_6365 = diff_ckp(n63, n65)
    with open("f136200.txt", "wt+") as fil:
        for item in diff_6365:
            if not "g2pa" in item:
                fil.write(item+"\n")