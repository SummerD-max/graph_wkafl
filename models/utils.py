import torch

from models import DGI


def get_DGI_model_optimizer():
    # hyperparameter
    hid_units = 256
    activation_fc = 'relu'
    lr = 0.001
    weight_decay = 0.0001
    ft_size = 3703
    # define model and optimizer
    model = DGI(n_in=ft_size, n_h=hid_units, activation=activation_fc)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, optimizer
