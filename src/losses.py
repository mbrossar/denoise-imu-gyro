import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
import matplotlib.pyplot as plt


class BaseLoss(torch.nn.Module):

    def __init__(self, min_N, max_N, dt):
        super().__init__()
        # windows sizes
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        # sampling time
        self.dt = dt # (s)


class GyroLoss(BaseLoss):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__(min_N, max_N, dt)
        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        self.weight = torch.ones(1, 1,
            self.min_train_freq).cuda()/self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w*self.sl(rs/self.huber,
            torch.zeros_like(rs))*(self.huber**2)
        return loss

    def forward_with_rotation_matrices(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternions(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        Xs = SO3.qexp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs[:,2])/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternion_mask(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        rs = rs[masks[:, self.N0:].squeeze(2) == 1]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss
