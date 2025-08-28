import numpy as np
import torch
from scipy.optimize import minimize, Bounds, minimize_scalar

#用于多任务相关，比如使用CAGrad
# class CAGrad:
#     def __init__(self, optimizer, alpha=0.5, rescale=1):
#         """
#         CAGrad 多任务优化器封装（仅作用于 optimizer 管理的参数）
#         """
#         self.optimizer = optimizer
#         self.alpha = alpha
#         self.rescale = rescale
#
#         # 获取设备（假设所有参数在同一设备）
#         self.device = self._get_device()
#
#     def _get_device(self):
#         for group in self.optimizer.param_groups:
#             for param in group['params']:
#                 return param.device
#         return torch.device("cpu")
#
#     def _get_all_params(self):
#         """仅获取 optimizer 管理的所有参数"""
#         params = []
#         for group in self.optimizer.param_groups:
#             params.extend(group['params'])
#         return [p for p in params if p.requires_grad]
#
#     def _gather_grads(self, losses):
#         grads = []
#         length = len(losses)
#         for idx, loss in enumerate(losses):
#             self.optimizer.zero_grad()
#             if idx < length-1:
#                 loss.backward(retain_graph=True)
#             else:
#                 loss.backward()
#             grad_list = []
#             for param in self._get_all_params():
#                 if param.grad is not None and param.requires_grad:
#                     grad_cur = param.grad.detach().clone()
#                     grad_list.append(grad_cur.view(-1))
#             grads.append(torch.cat(grad_list))
#         grads = torch.stack(grads, dim=1)  # shape: [total_params, num_tasks]
#         return grads
#
#     def _compute_direction(self, grads):
#         GG = grads.t().mm(grads).cpu()
#         g0_norm = (GG.mean() + 1e-8).sqrt()
#
#         num_tasks = grads.size(1)
#         x_start = np.ones(num_tasks) / num_tasks
#         bnds = tuple((0, 1) for _ in range(num_tasks))
#         cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
#         A = GG.numpy()
#         b = x_start.copy()
#         c = (self.alpha * g0_norm + 1e-8).item()
#
#         def objfn(x):
#             x = x.reshape(1, -1)
#             return (x @ A @ b.reshape(-1, 1) + c * np.sqrt(x @ A @ x.T + 1e-8)).sum()
#
#         res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
#         w = torch.tensor(res.x, dtype=torch.float32, device=self.device)
#         gw = (grads * w.view(1, -1)).sum(1)
#         gw_norm = gw.norm()
#         lmbda = c / (gw_norm + 1e-8)
#
#         g_final = grads.mean(1) + lmbda * gw
#         if self.rescale == 1:
#             g_final = g_final / (1 + self.alpha**2)
#         elif self.rescale != 0:
#             g_final = g_final / (1 + self.alpha)
#
#         return g_final.detach()
#
#     def _assign_grads_to_params(self, g_final):
#         offset = 0
#         for param in self._get_all_params():
#             if param.grad is not None :
#                 num_param = param.numel()
#                 param.grad = g_final[offset: offset + num_param].view_as(param).clone()
#                 offset += num_param
#
#     def backward(self, losses):
#         grads = self._gather_grads(losses)
#         g_final = self._compute_direction(grads)
#         self._assign_grads_to_params(g_final)

import numpy as np
import torch
from scipy.optimize import minimize

class CAGrad:
    def __init__(self,config,model, optimizer, alpha=0.5, rescale=1, use_dwa=False, T=2.0):
        self.config = config
        self.tooth_model = model
        self.optimizer = optimizer
        self.alpha = alpha
        self.rescale = rescale
        self.use_dwa = use_dwa
        self.T = T  # 温度参数

        #self.multi_task_for_oneTooth=config.multi_task_for_oneTooth
        #历史损失集合
        #1、总损失，把单牙齿损失加在一起，和牙弓、拥挤度损失并列
        self.sum_loss_history = []  # 每次记录一组任务的 loss 值（tensor），维度 [num_tasks]
        self.oneTooth_loss_history = []
        self.device = self._get_device()

    def _get_device(self):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                return param.device
        return torch.device("cpu")

    def _get_share_params_A(self):
        #获取模型的共享参数A部分，即图像信息编码器，同时进行了拥挤度分类以及宽度推理的fov_encoder的encoder部分
        module_A=self.tooth_model.get_shared_modules_A()
        params = []
        for k,v in module_A.named_parameters():
            if v.requires_grad:
                params.append(v)
        return params
        # for group in self.optimizer.param_groups:
        #     params.extend(group['params'])
        # return [p for p in params if p.requires_grad]
    def _get_share_params_B(self):
        param_B=[]
        shared_param_A=self._get_share_params_A()
        shared_params_set_A = set(shared_param_A)
        for k,v in self.tooth_model.named_parameters():
            if v not in shared_params_set_A and v.requires_grad:
                param_B.append(v)
        return param_B

    def _get_all_train_params(self):
        """仅获取 optimizer 管理的所有参数"""
        params = []
        for group in self.optimizer.param_groups:
            params.extend(group['params'])
        return [p for p in params if p.requires_grad]

    def _gather_grads(self,losses):
        param_train = self._get_all_train_params()


        grads = []

        length = len(losses)
        # if self.config.distributed:
        for idx, loss in enumerate(losses):
            self.optimizer.zero_grad()
            if idx < length - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            grad_list = []
            for param in param_train:
                if param.requires_grad:
                    if param.grad is not None:
                        grad_cur = param.grad.detach().clone()
                    else:
                        grad_cur = torch.zeros_like(param,device=param.device)
                    grad_list.append(grad_cur.view(-1))
            grads.append(torch.cat(grad_list))
        grads = torch.stack(grads, dim=1)
        # shape: [total_params, num_tasks]
        return grads



    def _gather_grads_share(self, losses):
        param_A=self._get_share_params_A()
        param_B=self._get_share_params_B()

        grads_A = []
        grads_B = []
        length = len(losses)
        #if self.config.distributed:
        for idx, loss in enumerate(losses):
            self.optimizer.zero_grad()
            if idx < length - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            grad_list_A = []
            for param in param_A:
                if param.grad is not None and param.requires_grad:
                    grad_cur = param.grad.detach().clone()
                    grad_list_A.append(grad_cur.view(-1))
            grads_A.append(torch.cat(grad_list_A))

            grad_list_B = []
            for param in param_B:
                if param.grad is not None and param.requires_grad:
                    grad_cur = param.grad.detach().clone()
                    grad_list_B.append(grad_cur.view(-1))
            grads_B.append(torch.cat(grad_list_B))
        grads_A = torch.stack(grads_A,dim=1)
        grads_B = torch.stack(grads_B,dim=1)
          # shape: [total_params, num_tasks]
        return grads_A,grads_B

    def _compute_direction(self,grads):
        GG = grads.t().mm(grads).cpu()
        g0_norm = (GG.mean() + 1e-8).sqrt()

        num_tasks = grads.size(1)
        x_start = np.ones(num_tasks) / num_tasks
        bnds = tuple((0, 1) for _ in range(num_tasks))
        cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (self.alpha * g0_norm + 1e-8).item()

        def objfn(x):
            x = x.reshape(1, -1)
            return (x @ A @ b.reshape(-1, 1) + c * np.sqrt(x @ A @ x.T + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w = torch.tensor(res.x, dtype=torch.float32, device=self.device)
        gw = (grads * w.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)

        g_final = grads.mean(1) + lmbda * gw
        if self.rescale == 1:
            g_final = g_final / (1 + self.alpha ** 2)
        elif self.rescale != 0:
            g_final = g_final / (1 + self.alpha)
        return g_final.detach()

    def _assign_grads_to_params(self, g_final):
        offset = 0
        all_params = self._get_all_train_params()
        for param in all_params:
            if param.requires_grad:
                num_param = param.numel()
                param.grad = g_final[offset: offset + num_param].view_as(param).clone()
                offset += num_param



    def _compute_dwa_weights(self,loss_history):
        if len(loss_history) < 2:
            num_tasks = loss_history[-1].numel()
            return torch.ones(num_tasks, device=self.device) / num_tasks

        loss_history = loss_history[-2:]
        L_t = loss_history[-1]
        L_t_1 = loss_history[-2]

        # 稳定化比值
        r = L_t / (L_t_1 + 1e-8)
        r = torch.clamp(r, min=1e-2, max=100.0)

        # 计算 softmax 权重
        w = torch.exp(r / self.T)
        w = len(r) * w / (w.sum() + 1e-8)  # 保持和为任务数
        return w.detach()

    def backward(self, losses):

        if self.use_dwa:
            #如果不计算单个牙齿的cagrad，直接合并了
            if not self.config.cagrad_for_oneTooth and self.config.loss_for_oneTooth:
                oneTooth_loss_tensor= torch.stack([losses[i].detach() for i in range(12)])
                self.oneTooth_loss_history.append(oneTooth_loss_tensor)
                one_tooth_weights = self._compute_dwa_weights(self.oneTooth_loss_history)
                losses_one_tooth=[w * l for w, l in zip(one_tooth_weights, losses[0:12])]
                one_tooth_sum=sum(losses_one_tooth)
                new_losses=losses[12:]
                new_losses.insert(0, one_tooth_sum)
                losses = new_losses


        losses_tensor = torch.stack([loss.detach() for loss in losses])
        self.sum_loss_history.append(losses_tensor)


        if self.use_dwa:
            weights = self._compute_dwa_weights(self.sum_loss_history)
            losses = [w * l for w, l in zip(weights, losses)]

        grads = self._gather_grads(losses)
        g_final = self._compute_direction(grads)
        self._assign_grads_to_params(g_final)

        #

