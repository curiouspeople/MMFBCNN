import torch
import torch.nn as nn
from MMBFCNN.spd import SPDTransform, SPDTangentSpace, SPDRectified
from torch.nn.functional import elu


class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cuda')

    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x @ x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov + (1e-5 * identity)
        return cov


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

    def patch_len(self, n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(AttentionManifold, self).__init__()

        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out)
        self.k_trans = SPDTransform(self.d_in, self.d_out)
        self.v_trans = SPDTransform(self.d_in, self.d_out)

    def tensor_log(self, t):  # 4dim
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_exp(self, t):  # 4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)

    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _ = torch.svd(inner_multi)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov  # [bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)  # patch:[b, #patch, c, c]
        q_list = [];
        k_list = [];
        v_list = []
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)

        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                 K_expand.shape[4])

        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)  # now row is c.c.

        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape


class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
                self.__class__.__name__
                + "("
                + "expression="
                + str(expression_str)
                + ")"
        )


def identity(x):
    """
    No activation function
    """
    return x


def _transpose1(x):
    return x.permute(0, 3, 2, 1)


def _transpose2(x):
    return x.permute(0, 1, 3, 2)


class SELayer(nn.Module):
    """
    the Squeeze and Excitation layer, defined by formula (4)(5) in the paper

    Parameters
    ----------
    channel: the input channel number
    reduction: the reduction ratio r
    """

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # nn.Sigmoid()
            nn.Softmax()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class MMBFCNN_bci(nn.Module):
    def __init__(self, epochs,
                 in_chans=22,
                 n_classes=4,
                 reduction_ratio=8,
                 conv_stride=1,
                 pool_stride=2,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.4,
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.reduction_ratio = reduction_ratio
        # FE
        # bs, 1, channel, sample 空间卷积
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample 时间卷积
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(20)

        # 时空卷积
        self.transpose1 = Expression(_transpose1)
        self.conv_time = nn.Conv2d(1, 25, (11, 1), stride=1)
        self.conv_spatial = nn.Conv2d(25, 25, (1, self.in_chans), stride=1, bias=not self.batch_norm)
        self.bn0 = nn.BatchNorm2d(25, momentum=self.batch_norm_alpha, affine=True)
        self.conv_nonlinear = Expression(elu)  # return x*x
        self.first_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.pool_nonlinear = Expression(identity)  # th.log(th.clamp(x, min=eps))

        # 深度时间卷积
        # the Spatio-Temporal Block

        # self.transpose1 = Expression(_transpose1)
        self.transpose2 = Expression(_transpose2)
        self.first_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.pool_nonlinear = Expression(identity)  # th.log(th.clamp(x, min=eps))
        # the 1-st Temporal Conv Unit
        self.dtdrop1 = nn.Dropout(p=self.drop_prob)
        self.dtconv1 = nn.Conv2d(25, 100, (11, 1), stride=(conv_stride, 1), bias=not self.batch_norm)
        self.dtbn1 = nn.BatchNorm2d(100, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.dtconv_nonlinear1 = Expression(elu)
        self.dtpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.dtpool_nonlinear1 = Expression(identity)

        # the 2-nd Temporal Conv Unit
        self.dtdrop2 = nn.Dropout(p=self.drop_prob)
        self.dtconv2 = nn.Conv2d(100, 100, (11, 1), stride=(conv_stride, 1), bias=not self.batch_norm)
        self.dtbn2 = nn.BatchNorm2d(100, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.dtconv_nonlinear2 = Expression(elu)
        self.dtpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.dtpool_nonlinear2 = Expression(identity)

        # the 3-rd Temporal Conv Unit
        self.dtdrop3 = nn.Dropout(p=self.drop_prob)
        self.dtconv3 = nn.Conv2d(100, 100, (11, 1), stride=(conv_stride, 1), bias=not self.batch_norm)
        self.dtbn3 = nn.BatchNorm2d(100, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.dtconv_nonlinear3 = Expression(elu)
        self.dtpool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.dtpool_nonlinear3 = Expression(identity)

        # The SEC Unit for Deep-Temporal features
        self.SElayer2 = SELayer(100, self.reduction_ratio)
        self.SEconv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(1, 3), stride=(1, 1),
                                 padding=(0, 3 // 2),
                                 groups=1, bias=True)
        self.SEbn2 = nn.BatchNorm2d(100)
        self.SEpooling2 = nn.AdaptiveAvgPool2d(output_size=(8, 1))

        self.elu = nn.ELU(inplace=True)

        self.SElayer1 = SELayer(100, self.reduction_ratio)
        self.SEbn1 = nn.BatchNorm2d(100)

        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(25, 25)
        self.ract2 = SPDRectified()

        # R2E
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(9 * 19 * epochs, 4, bias=True)

        self.conv1d = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool_1d = nn.AdaptiveAvgPool1d(output_size=100)

        # the Classifier
        self.conv_classifier = nn.Conv2d(100, self.n_classes, (9, 1), bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.squeeze_output = Expression(_squeeze_final_output)

    def forward(self, x):

        # 时空卷积
        # x = self.conv1(x)
        # x = self.Bn1(x)
        # x = self.conv2(x)
        # x = self.Bn2(x)
        # the Spatio-Temporal Block

        x = self.transpose1(x)

        x = self.conv_time(x)

        x = self.conv_spatial(x)

        x = self.bn0(x)
        x = self.conv_nonlinear(x)

        # 深度时间卷积

        x_dt = x
        # x_dt = self.transpose2(x_dt)
        # the Deep-Temporal Convolution Block

        x_dt = self.first_pool(x_dt)

        x_dt = self.pool_nonlinear(x_dt)
        # the 1-st Temporal Conv Unit
        x_dt = self.dtconv_nonlinear1(self.dtbn1(self.dtconv1(self.dtdrop1(x_dt))))
        x_dt = self.dtpool_nonlinear1(self.dtpool1(x_dt))
        # the 2-nd Temporal Conv Unit
        x_dt = self.dtconv_nonlinear2(self.dtbn2(self.dtconv2(self.dtdrop2(x_dt))))
        x_dt = self.dtpool_nonlinear2(self.dtpool2(x_dt))
        # the 3-rd Temporal Conv Unit
        x_dt = self.dtconv_nonlinear3(self.dtbn3(self.dtconv3(self.dtdrop3(x_dt))))
        x_dt = self.dtpool3(x_dt)
        # The SEC Unit for Deep-Temporal features
        x_dt = self.SElayer2(x_dt)
        x_dt = self.elu(self.SEbn2(self.SEconv2(x_dt)))
        x_dt = self.SEpooling2(x_dt)
        # torch.Size([1, 100, 8, 1])



        # torch.Size([1, 20, 1, 439])
        # torch.Size([72, 25, 1115, 1])
        x = self.transpose2(x)

        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        x = self.tangent(x)

        # torch.Size([3, 171])
        x = x.view(shape[0], shape[1], -1)
        # torch.Size([1, 3, 171])
        x = self.conv1d(x)
        x = self.adaptive_pool_1d(x)
        x = x.unsqueeze(2)

        x = self.transpose1(x)
        x = self.SElayer1(x)


        x = torch.cat((x, x_dt), 2)

        x = self.conv_classifier(x)

        x = self.softmax(x)
        x = self.squeeze_output(x)
        return x

class MMBFCNN_bci_2b(nn.Module):
    def __init__(self, epochs,
                 in_chans=3,
                 n_classes=2,
                 reduction_ratio=8,
                 conv_stride=1,
                 pool_stride=2,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.4,
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.reduction_ratio = reduction_ratio
        # FE
        # bs, 1, channel, sample 空间卷积
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample 时间卷积
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(20)

        # 时空卷积
        self.transpose1 = Expression(_transpose1)
        self.conv_time = nn.Conv2d(1, 25, (11, 1), stride=1)
        self.conv_spatial = nn.Conv2d(25, 25, (1, self.in_chans), stride=1, bias=not self.batch_norm)
        self.bn0 = nn.BatchNorm2d(25, momentum=self.batch_norm_alpha, affine=True)
        self.conv_nonlinear = Expression(elu)  # return x*x
        self.first_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.pool_nonlinear = Expression(identity)  # th.log(th.clamp(x, min=eps))

        # 深度时间卷积
        # the Spatio-Temporal Block

        # self.transpose1 = Expression(_transpose1)
        self.transpose2 = Expression(_transpose2)
        self.first_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.pool_nonlinear = Expression(identity)  # th.log(th.clamp(x, min=eps))
        # the 1-st Temporal Conv Unit
        self.dtdrop1 = nn.Dropout(p=self.drop_prob)
        self.dtconv1 = nn.Conv2d(25, 100, (11, 1), stride=(conv_stride, 1), bias=not self.batch_norm)
        self.dtbn1 = nn.BatchNorm2d(100, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.dtconv_nonlinear1 = Expression(elu)
        self.dtpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.dtpool_nonlinear1 = Expression(identity)

        # the 2-nd Temporal Conv Unit
        self.dtdrop2 = nn.Dropout(p=self.drop_prob)
        self.dtconv2 = nn.Conv2d(100, 100, (11, 1), stride=(conv_stride, 1), bias=not self.batch_norm)
        self.dtbn2 = nn.BatchNorm2d(100, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.dtconv_nonlinear2 = Expression(elu)
        self.dtpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.dtpool_nonlinear2 = Expression(identity)

        # the 3-rd Temporal Conv Unit
        self.dtdrop3 = nn.Dropout(p=self.drop_prob)
        self.dtconv3 = nn.Conv2d(100, 100, (11, 1), stride=(conv_stride, 1), bias=not self.batch_norm)
        self.dtbn3 = nn.BatchNorm2d(100, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.dtconv_nonlinear3 = Expression(elu)
        self.dtpool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))
        self.dtpool_nonlinear3 = Expression(identity)

        # The SEC Unit for Deep-Temporal features
        self.SElayer2 = SELayer(100, self.reduction_ratio)
        self.SEconv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(1, 3), stride=(1, 1),
                                 padding=(0, 3 // 2),
                                 groups=1, bias=True)
        self.SEbn2 = nn.BatchNorm2d(100)
        self.SEpooling2 = nn.AdaptiveAvgPool2d(output_size=(8, 1))

        self.elu = nn.ELU(inplace=True)

        self.SElayer1 = SELayer(100, self.reduction_ratio)
        self.SEbn1 = nn.BatchNorm2d(100)

        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(25, 25)
        self.ract2 = SPDRectified()

        # R2E
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(9 * 19 * epochs, 4, bias=True)

        self.conv1d = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool_1d = nn.AdaptiveAvgPool1d(output_size=100)

        # the Classifier
        self.conv_classifier = nn.Conv2d(100, self.n_classes, (9, 1), bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.squeeze_output = Expression(_squeeze_final_output)

    def forward(self, x):

        # 时空卷积
        # x = self.conv1(x)
        # x = self.Bn1(x)
        # x = self.conv2(x)
        # x = self.Bn2(x)
        # the Spatio-Temporal Block

        x = self.transpose1(x)

        x = self.conv_time(x)

        x = self.conv_spatial(x)

        x = self.bn0(x)
        x = self.conv_nonlinear(x)

        # 深度时间卷积

        x_dt = x
        # x_dt = self.transpose2(x_dt)
        # the Deep-Temporal Convolution Block

        x_dt = self.first_pool(x_dt)

        x_dt = self.pool_nonlinear(x_dt)
        # the 1-st Temporal Conv Unit
        x_dt = self.dtconv_nonlinear1(self.dtbn1(self.dtconv1(self.dtdrop1(x_dt))))
        x_dt = self.dtpool_nonlinear1(self.dtpool1(x_dt))
        # the 2-nd Temporal Conv Unit
        x_dt = self.dtconv_nonlinear2(self.dtbn2(self.dtconv2(self.dtdrop2(x_dt))))
        x_dt = self.dtpool_nonlinear2(self.dtpool2(x_dt))
        # the 3-rd Temporal Conv Unit
        x_dt = self.dtconv_nonlinear3(self.dtbn3(self.dtconv3(self.dtdrop3(x_dt))))
        x_dt = self.dtpool3(x_dt)
        # The SEC Unit for Deep-Temporal features
        x_dt = self.SElayer2(x_dt)
        x_dt = self.elu(self.SEbn2(self.SEconv2(x_dt)))
        x_dt = self.SEpooling2(x_dt)
        # torch.Size([1, 100, 8, 1])



        # torch.Size([1, 20, 1, 439])
        # torch.Size([72, 25, 1115, 1])
        x = self.transpose2(x)

        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        x = self.tangent(x)

        # torch.Size([3, 171])
        x = x.view(shape[0], shape[1], -1)
        # torch.Size([1, 3, 171])
        x = self.conv1d(x)
        x = self.adaptive_pool_1d(x)
        x = x.unsqueeze(2)

        x = self.transpose1(x)
        x = self.SElayer1(x)


        x = torch.cat((x, x_dt), 2)

        x = self.conv_classifier(x)

        x = self.softmax(x)

        x = self.squeeze_output(x)

        return x


class MMBFCNN_mamem(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 125, (8, 1))
        self.Bn1 = nn.BatchNorm2d(125)
        # bs, 8, 1, sample
        self.conv2 = nn.Conv2d(125, 15, (1, 36), padding=(0, 18))
        self.Bn2 = nn.BatchNorm2d(15)

        # E2R
        self.ract1 = E2R(epochs)
        # riemannian part
        self.att2 = AttentionManifold(15, 12)
        self.ract2 = SPDRectified()
        # R2E
        self.tangent = SPDTangentSpace(12)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(6 * 13 * epochs, 5, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)

        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class MMBFCNN_cha(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (56, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 56, 1, sample
        self.conv2 = nn.Conv2d(22, 16, (1, 64), padding=(0, 32))
        self.Bn2 = nn.BatchNorm2d(16)

        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(16, 8)
        self.ract2 = SPDRectified()

        # R2E
        self.tangent = SPDTangentSpace(8)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(4 * 9 * epochs, 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)


        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)

        x = self.tangent(x)

        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = MMBFCNN_bci(3, 4)
    total_params = sum(p.numel() for p in model.parameters())
