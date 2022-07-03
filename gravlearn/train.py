import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_sampler import TripletDataset, nGramSampler, FrequencyBasedSampler
from .losses import TripletLoss
from .metrics import DistanceMetrics


def train(
    model,
    seqs,
    window_length,
    dist_metric,
    bags=None,
    batch_size=256,
    device=None,
    epochs=1,
    checkpoint=10000,
    outputfile=None,
    context_window_type="double",
    **params
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
    # Set up dataset
    #
    pos_sampler = nGramSampler(
        window_length=window_length, context_window_type=context_window_type
    )
    neg_sampler = FrequencyBasedSampler(gamma=1)
    pos_sampler.fit(seqs)
    neg_sampler.fit(seqs)
    dataset = TripletDataset(
        epochs=epochs, pos_sampler=pos_sampler, neg_sampler=neg_sampler
    )

    dataloader = DataLoader(
        dataset,
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
    loss_func = TripletLoss(embedding=model, dist_metric=dist_metric)

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = AdamW(focal_params)

    pbar = tqdm(enumerate(dataloader), miniters=100, total=len(dataloader))
    for it, (p1, p2, n1) in pbar:

        # clear out the gradient
        focal_params = filter(lambda p: p.requires_grad, model.parameters())
        for param in focal_params:
            param.grad = None

        # Convert to bags if bags are given
        if bags is not None:
            p1, p2, n1 = bags[p1], bags[p2], bags[n1]
        else:
            p1, p2, n1 = p1.to(device), p2.to(device), n1.to(device)

        # compute the loss
        loss = loss_func(p1, p2, n1)

        # backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(focal_params, 1)

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
