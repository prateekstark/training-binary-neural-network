import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def calc_entropy(input_tensor):
    lsm = torch.nn.LogSoftmax(dim=0).to("cuda")
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy


if __name__ == "__main__":
    index = 2

    tensor = torch.load("lamda_{}.pth".format(index), map_location=torch.device("cpu"))
    p = torch.sigmoid(2 * tensor)
    shape = tensor.shape[0]
    bins = 200
    hist = torch.histc(p, bins=bins, min=0, max=1).cpu().detach().numpy()
    x = []
    start = 0
    for i in range(bins):
        x.append(start)
        start += 1.0 / bins
    print(calc_entropy(tensor))
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=1)
    plt.plot(x, hist * 100 / shape)
    plt.xlabel("p(w=1)", size=18)
    plt.ylabel("% of weights", size=18)
    plt.savefig("after_task_{}.png".format(index), dpi=300)
    plt.show()


# plt.xlabel('Epochs', size=22)
# plt.ylabel('Accuracy (%)', size=22)
# plt.legend(loc=4, prop={'size': 22})
# plt.savefig('CIFAR100.png', dpi=300)
