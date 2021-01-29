import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


if __name__ == "__main__":
    tensor = torch.load("lamda_1.pth", map_location=torch.device("cpu"))
    p = torch.sigmoid(2 * tensor).cpu().detach().numpy()
    density = st.gaussian_kde(p)
    n, x, _ = plt.hist(p, bins=np.linspace(0, 1, 10), histtype=u"step", density=True)
    plt.plot(x, density(x))
    plt.show()
