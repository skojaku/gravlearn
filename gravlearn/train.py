import torch
from scipy import sparse
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_sampler import QuadletSampler
from .losses import QuadletLoss, TripletLoss
from .metrics import DistanceMetrics


def train(
    model,
    seqs,
    window_length,
    dist_metric,
    bags=None,
    batch_size=256,
    device=None,
    data_buffer_size=100000,
    epochs=1,
    checkpoint=10000,
    outputfile=None,
    share_center=False,
    train_by_triplet=False,
    context_window_type="double",
):
    # Set the device parameter if not specified
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # flag up "model.normalize" if the dist metric is scale invariant
    if DistanceMetrics.is_scale_invariant(dist_metric):
        model.normalize = True
    else:  # if it is scale variant, then mute the scale parameter
        model.scale.requires_grad = False

    #
    # Set up data sampler
    #
    sampler = QuadletSampler(
        seqs=seqs,
        window_length=window_length,
        buffer_size=data_buffer_size,
        epochs=epochs,
        share_center=share_center,
        context_window_type=context_window_type,
    )

    dataloader = DataLoader(
        sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    #
    # Set up the model
    #
    model.train()
    model = model.to(device)

    #
    # Set up the loss function
    #
    if train_by_triplet:
        loss_func = TripletLoss(embedding=model, dist_metric=dist_metric)
    else:
        loss_func = QuadletLoss(embedding=model, dist_metric=dist_metric)

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    # optim = Adam(focal_params, lr=0.003)
    optim = AdamW(focal_params)

    pbar = tqdm(enumerate(dataloader), miniters=100, total=len(dataloader))
    for it, (p1, p2, n1, n2) in pbar:

        # clear out the gradient
        focal_params = filter(lambda p: p.requires_grad, model.parameters())
        for param in focal_params:
            param.grad = None

        # Convert to bags if bags are given
        if bags is not None:
            p1, p2, n1, n2 = bags[p1], bags[p2], bags[n1], bags[n2]
        else:
            p1, p2, n1, n2 = p1.to(device), p2.to(device), n1.to(device), n2.to(device)

        # compute the loss
        if train_by_triplet:
            loss = loss_func(p1, p2, n2)
        else:
            loss = loss_func(p1, p2, n1, n2)

        # backpropagate
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(focal_params, 1)

        # update the parameters
        optim.step()

        pbar.set_postfix(loss=loss.item())

        if (it + 1) % checkpoint == 0:
            if outputfile is not None:
                torch.save(model.state_dict(), outputfile)

    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model
