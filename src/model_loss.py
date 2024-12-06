import torch.nn.functional as F
import torch

def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:
    n = logits.shape[1]
    labels = torch.arange(n)
    logits = logits.to("cpu")
    #calculate image and text loss
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    loss = (loss_i + loss_t)/2
    return loss

def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc