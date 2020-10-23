import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BiNNOptimizer(Optimizer):
    def __init__(
        self,
        model,
        N,
        mini_batch_size,
        learning_rate,
        temperature,
        initialize_lambda,
        beta,
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
            mini_batch_size=mini_batch_size,
            N=N,
            learning_rate=learning_rate,
            temperature=temperature,
            beta=beta,
        )
        super(Binn_optimizer, self).__init__(model.parameters(), default_dict)

        ## Natural parameter prior lambda = 0

        self.train_modules = []
        get_train_modules(model)

        for group in self.param_groups:
            group["lr"] = learning_rate
            group["beta"] = beta
            p = parameters_to_vector(group["params"])

            # Initialization lamda  between -10 and 10
            # Convex combination
            theta1 = torch.randint(0, 2, p.size())
            self.state["lambda"] = theta1 * (initialize_lambda) + (1 - theta1) * (
                -initialize_lambda
            )

            self.state["mu"] = torch.tanh(self.state["lambda"])
            # Lambda_0 initialized as 0
            self.state["lambda_prior"] = torch.zeros_like(p)
            self.state["step"] = 0
            self.state["temperature"] = temperature
            self.state["momentum"] = torch.zeros_like(p)

            break

    def get_train_modules(self, model):
        """
        To get all the modules which have trainiable parameters.
        """
        if len(list(model.children())) == 0:
            if len(list(model.parameters())) != 0:
                self.train_modules.append(model)
        else:
            for sub_module in list(model.children()):
                get_train_modules(sub_modules)

    def step(self, closure=None):

        loss = None
        pred = None

        if closure is not None:
            pass
        else:
            raise RuntimeError(
                "Something is wrong in step function of optimizer class, Please Check!"
            )

        self.state["step"] += 1

        lr = self.param_groups[0]["lr"]
        beta = self.param_groups["beta"]
        parameters = self.param_groups[0]["params"]
        momentum = self.state["momentum"]
        N = self.defaults["N"]
        M = self.defaults["mini_batch_size"]
        mu = self.state["mu"]
        lamda = self.state["lambda"]
        temperature = self.state["temperature"]

        grad = torch.zeros_like(lamda)

        loss_list = []
        if M <= 0:
            relaxed_w = torch.tanh(self.state["lambda"])
            vector_to_parameters(relaxed_w, parameters)

            loss = closure()

            g = parameters_to_vector(torch.autograd.grad(loss, parameters)).detach()
            grad = N * g

            loss_list.append(loss.detach())

        else:
            for num in range(M):

                epsilon = torch.rand_like(lamda)
                delta = 0.5 * torch.log(epsilon / (1 - epsilon))
                relaxed_w = torch.tanh((lamda + delta) / temperature)
                vector_to_parameters(relaxed_w, parameters)

                loss = closure()

                g = parameters_to_vector(torch.autograd.grad(loss, parameters)).detach()
                s = (N / temperature) * ((1 - relaxed_w * relaxed_w) / (1 - mu * mu))
                grad.add_(s * g)

                loss_list.append(loss.detach())

        grad.mul_(1 / M)

        self.state["momentum"] = beta * self.state["momentum"] + (1 - beta) * (
            grad + self.state["lambda"]
        )  ## P

        bias_correction1 = 1 - beta ** self.state["step"]

        self.state["lambda"] = (
            self.state["lambda"] - lr * self.state["momentum"] / bias_correction1
        )
        self.state["mu"] = torch.tanh(self.state["lambda"])

        loss = torch.mean(torch.stack(loss_list))

        return loss
