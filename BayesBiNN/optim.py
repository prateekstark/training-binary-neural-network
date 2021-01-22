import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BiNNOptimizer(Optimizer):
    def __init__(
        self,
        model,
        train_set_size,
        N=5,
        learning_rate=1e-9,
        temperature=1e-10,
        initialize_lambda=10,
        beta=0.99,
    ):
        """
        For torch's Optimizer class
            Arguments:
            params (iterable): an iterable of :class:`torch.Tensor` s or
                :class:`dict` s. Specifies what Tensors should be optimized.
            defaults: (dict): a dict containing default values of optimization
                options (used when a parameter group doesn't specify them).
        """
        default_dict = dict(
            N=N,
            learning_rate=learning_rate,
            temperature=temperature,
            beta=beta,
            train_set_size=train_set_size,
        )

        super(BiNNOptimizer, self).__init__(model.parameters(), default_dict)

        ## Natural parameter prior lambda = 0

        self.train_modules = []
        self.get_train_modules(model)

        self.param_groups[0]["lr"] = learning_rate
        self.param_groups[0]["beta"] = beta
        p = parameters_to_vector(self.param_groups[0]["params"])

        # Initialization lamda  between -10 and 10
        # Convex combination
        theta1 = torch.randint_like(p, 2)
        self.state["lambda"] = (theta1 * initialize_lambda) - (
            (1 - theta1) * initialize_lambda
        )
        self.state["mu"] = torch.tanh(self.state["lambda"])
        self.state["momentum"] = torch.zeros_like(p)
        self.state["lambda_prior"] = torch.zeros_like(p)
        self.state["step"] = 0
        self.state["temperature"] = temperature

    def get_train_modules(self, model):
        """
        To get all the modules which have trainiable parameters.
        """
        if len(list(model.children())) == 0:
            if len(list(model.parameters())) != 0:
                self.train_modules.append(model)
        else:
            for sub_module in list(model.children()):
                self.get_train_modules(sub_module)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError(
                "Something is wrong in step function of optimizer class, Please Check!"
            )

        self.state["step"] += 1

        lr = self.param_groups[0]["lr"]
        parameters = self.param_groups[0]["params"]

        N = self.defaults["N"]
        beta = self.defaults["beta"]
        M = self.defaults["train_set_size"]

        mu = self.state["mu"]
        lamda = self.state["lambda"]

        temperature = self.defaults["temperature"]
        grad = torch.zeros_like(lamda)

        loss_list = []
        pred_list = []
        if N <= 0:
            relaxed_w = torch.tanh(self.state["lambda"])
            vector_to_parameters(relaxed_w, parameters)
            loss, pred = closure()
            pred_list.append(pred)
            loss_list.append(loss.detach())
            g_temp = torch.autograd.grad(loss, parameters)
            g = parameters_to_vector(g_temp).detach()
            grad = M * g
        else:
            for num in range(N):
                epsilon = torch.rand_like(mu)
                delta = torch.log(epsilon / (1 - epsilon)) / 2
                relaxed_w = torch.tanh((self.state["lambda"] + delta) / temperature)

                vector_to_parameters(relaxed_w, parameters)
                loss, pred = closure()
                pred_list.append(pred)
                loss_list.append(loss.detach())

                g = parameters_to_vector(torch.autograd.grad(loss, parameters)).detach()
                s = (
                    (1 - relaxed_w * relaxed_w + 1e-10)
                    / temperature
                    / (1 - self.state["mu"] * self.state["mu"] + 1e-10)
                )
                grad.add_(s * g)

            grad.mul_(M / N)

        self.state["momentum"] = beta * self.state["momentum"] + (1 - beta) * (
            grad + self.state["lambda"]
        )  ## P

        loss = torch.mean(torch.stack(loss_list))

        bias_correction1 = 1 - beta ** self.state["step"]

        self.state["lambda"] = (
            self.state["lambda"]
            - self.param_groups[0]["lr"] * self.state["momentum"] / bias_correction1
        )
        self.state["mu"] = torch.tanh(lamda)
        return loss, pred_list
