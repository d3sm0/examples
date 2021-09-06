import torch

import experiment_buddy


def main():
    experiment_buddy.register_defaults(dict(batch_size=32))
    tb = experiment_buddy.deploy(host="mila")
    linear = torch.nn.Linear(32,1)
    tb.add_object("model", linear, global_step=0)

if __name__ == '__main__':
    main()



