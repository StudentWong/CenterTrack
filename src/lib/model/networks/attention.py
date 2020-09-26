import torch
from torch import nn


class TimeAttention(torch.nn.Module):
    def __init__(self, opt, channel):
        super(TimeAttention, self).__init__()
        self.max_object = opt.K
        self.history_T = opt.History_T
        self.channel = channel
        self.cos_similiarty = nn.CosineSimilarity(dim=3)
        self.softmax = nn.Softmax(dim=1)
        self.fc_pr = nn.Linear(channel, channel)
        self.cnn_ft = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, pre_obj_ft, ft):
        #pre_obj_ft: N*T*O*C
        #ft: N*C*H*W
        assert len(pre_obj_ft.shape) == 4 and len(ft.shape) == 4
        assert pre_obj_ft.shape[0] == ft.shape[0]
        N = pre_obj_ft.shape[0]
        assert pre_obj_ft.shape[3] == ft.shape[1] and pre_obj_ft.shape[3] == self.channel
        assert pre_obj_ft.shape[1] == self.history_T
        assert pre_obj_ft.shape[2] == self.max_object
        H = ft.shape[2]
        W = ft.shape[3]

        # print(pre_obj_ft.shape)
        # print(ft.shape)
        assert pre_obj_ft.is_contiguous()
        pre_obj_ft_adj = self.fc_pr(
                    pre_obj_ft.view(N*self.history_T*self.max_object, self.channel)
                    )

        assert ft.is_contiguous()
        ft_adj = self.cnn_ft(ft)

        assert pre_obj_ft_adj.is_contiguous()
        pre_obj_ft_adj = pre_obj_ft_adj.view(N, self.history_T*self.max_object, self.channel)\
            .unsqueeze(1).expand(N, H*W, self.history_T*self.max_object, self.channel)
        assert ft_adj.is_contiguous()
        ft_adj = ft_adj.view(N, H*W, self.channel)\
            .unsqueeze(2).expand(N, H*W, self.history_T*self.max_object, self.channel)

        sim = self.cos_similiarty(pre_obj_ft_adj, ft_adj)

        att_mat = self.softmax(sim)
        assert att_mat.is_contiguous()
        att_mat_hw1 = att_mat.view(N, H, W, self.history_T*self.max_object).sum(dim=3, keepdim=True)
        att_mat_1hw = att_mat_hw1.permute((0, 3, 1, 2)).contiguous()
        assert att_mat_1hw.is_contiguous()
        ret = att_mat_1hw.expand_as(ft) * ft

        return ret

        # print(att_mat_hw1.shape)

        # print(att_mat.sum(dim=1))
        # print(sim.shape)
        # print(ft_adj.shape)