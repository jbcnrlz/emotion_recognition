import torch
import torch.nn as nn
from torch.nn import functional as F

torch_ver = torch.__version__[:3]

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.query_conv)
        kaiming_init(self.key_conv)
        kaiming_init(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Calculate(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Calculate, self).__init__()
        self.chanel_in = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.contiguous().view(m_batchsize, C, -1)
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        return attention


class CAM_Use(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Use, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                attention: B X C X C
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        proj_value = x.contiguous().view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class FeatureEnhance(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhance, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = nn.ModuleList()
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(self.depthLayers):
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_cals.append(CAM_Calculate(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))
            self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            #self.deform_convs.append(DeformableConv2d(in_channels=self.in_channels,
            #                                        out_channels=self.out_channels,
            #                                        kernel_size=3,
            #                                        padding=1))

    def forward(self, r_f, g_f, b_f, a_f):        

        for idx in range(self.depthLayers):

            r_sp_feat = self.pams[idx](r_f)
            g_sp_feat = self.pams[idx](g_f)
            b_sp_feat = self.pams[idx](b_f)
            a_sp_feat = self.pams[idx](a_f)

            r_attention = self.cam_cals[idx](r_f)
            g_attention = self.cam_cals[idx](g_f)
            b_attention = self.cam_cals[idx](b_f)
            a_attention = self.cam_cals[idx](a_f)

            rr_sc_feat = self.cam_uses[idx](r_f, r_attention)
            gg_sc_feat = self.cam_uses[idx](g_f, g_attention)
            bb_sc_feat = self.cam_uses[idx](b_f, b_attention)
            aa_sc_feat = self.cam_uses[idx](a_f, a_attention)

            rg_sc_feat = self.cam_uses[idx](r_f, g_attention)
            rb_sc_feat = self.cam_uses[idx](r_f, b_attention)
            ra_sc_feat = self.cam_uses[idx](r_f, a_attention)

            gr_sc_feat = self.cam_uses[idx](g_f, r_attention)
            gb_sc_feat = self.cam_uses[idx](g_f, b_attention)
            ga_sc_feat = self.cam_uses[idx](g_f, a_attention)

            bg_sc_feat = self.cam_uses[idx](b_f, g_attention)
            br_sc_feat = self.cam_uses[idx](b_f, r_attention)
            ba_sc_feat = self.cam_uses[idx](b_f, a_attention)

            ag_sc_feat = self.cam_uses[idx](a_f, g_attention)
            ab_sc_feat = self.cam_uses[idx](a_f, b_attention)
            ar_sc_feat = self.cam_uses[idx](a_f, r_attention)

            #z_f = z_sp_feat + zz_sc_feat + zx_sc_feat
            #x_f = x_sp_feat + xx_sc_feat + xz_sc_feat

            r_f = r_sp_feat + rr_sc_feat + rg_sc_feat + rb_sc_feat + ra_sc_feat
            g_f = g_sp_feat + gg_sc_feat + gr_sc_feat + gb_sc_feat + ga_sc_feat
            b_f = b_sp_feat + bb_sc_feat + br_sc_feat + bg_sc_feat + ba_sc_feat
            a_f = a_sp_feat + aa_sc_feat + ag_sc_feat + ab_sc_feat + ar_sc_feat

            r_f = self.deform_convs[idx](r_f)
            g_f = self.deform_convs[idx](g_f)
            b_f = self.deform_convs[idx](b_f)
            a_f = self.deform_convs[idx](a_f)

            #z_f = self.deform_convs[idx](z_f)
            #x_f = self.deform_convs[idx](x_f)

        return r_f, g_f, b_f, a_f

class FeatureEnhanceNoCross(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceNoCross, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = PAM_Module(self.in_channels)
        self.deform_convs = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, r_f):
        r_sp_feat = self.pams(r_f)
        r_f = self.deform_convs(r_f * r_sp_feat)
        return r_f

class FeatureEnhanceDepthDI(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceDepthDI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = nn.ModuleList()
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(self.depthLayers):
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_cals.append(CAM_Calculate(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))
            self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            #self.deform_convs.append(DeformableConv2d(in_channels=self.in_channels,
            #                                        out_channels=self.out_channels,
            #                                        kernel_size=3,
            #                                        padding=1))

    def forward(self, r_f, g_f, b_f, a_f,d_f):        

        for idx in range(self.depthLayers):

            r_sp_feat = self.pams[idx](r_f)
            g_sp_feat = self.pams[idx](g_f)
            b_sp_feat = self.pams[idx](b_f)
            a_sp_feat = self.pams[idx](a_f)
            d_sp_feat = self.pams[idx](d_f)

            r_attention = self.cam_cals[idx](r_f)
            g_attention = self.cam_cals[idx](g_f)
            b_attention = self.cam_cals[idx](b_f)
            a_attention = self.cam_cals[idx](a_f)
            d_attention = self.cam_cals[idx](d_f)

            rr_sc_feat = self.cam_uses[idx](r_f, r_attention)
            gg_sc_feat = self.cam_uses[idx](g_f, g_attention)
            bb_sc_feat = self.cam_uses[idx](b_f, b_attention)
            aa_sc_feat = self.cam_uses[idx](a_f, a_attention)
            dd_sc_feat = self.cam_uses[idx](d_f, d_attention)

            rg_sc_feat = self.cam_uses[idx](r_f, g_attention)
            rb_sc_feat = self.cam_uses[idx](r_f, b_attention)
            ra_sc_feat = self.cam_uses[idx](r_f, a_attention)
            rd_sc_feat = self.cam_uses[idx](r_f, d_attention)

            gr_sc_feat = self.cam_uses[idx](g_f, r_attention)
            gb_sc_feat = self.cam_uses[idx](g_f, b_attention)
            ga_sc_feat = self.cam_uses[idx](g_f, a_attention)
            gd_sc_feat = self.cam_uses[idx](g_f, d_attention)

            bg_sc_feat = self.cam_uses[idx](b_f, g_attention)
            br_sc_feat = self.cam_uses[idx](b_f, r_attention)
            ba_sc_feat = self.cam_uses[idx](b_f, a_attention)
            bd_sc_feat = self.cam_uses[idx](b_f, d_attention)

            ag_sc_feat = self.cam_uses[idx](a_f, g_attention)
            ab_sc_feat = self.cam_uses[idx](a_f, b_attention)
            ar_sc_feat = self.cam_uses[idx](a_f, r_attention)
            ad_sc_feat = self.cam_uses[idx](a_f, d_attention)

            dr_sc_feat = self.cam_uses[idx](d_f, r_attention)
            dg_sc_feat = self.cam_uses[idx](d_f, g_attention)
            db_sc_feat = self.cam_uses[idx](d_f, b_attention)
            da_sc_feat = self.cam_uses[idx](d_f, a_attention)

            r_f = r_sp_feat + rr_sc_feat + rg_sc_feat + rb_sc_feat + ra_sc_feat + rd_sc_feat
            g_f = g_sp_feat + gg_sc_feat + gr_sc_feat + gb_sc_feat + ga_sc_feat + gd_sc_feat
            b_f = b_sp_feat + bb_sc_feat + br_sc_feat + bg_sc_feat + ba_sc_feat + bd_sc_feat
            a_f = a_sp_feat + aa_sc_feat + ag_sc_feat + ab_sc_feat + ar_sc_feat + ad_sc_feat
            d_f = d_sp_feat + dd_sc_feat + dr_sc_feat + dg_sc_feat + da_sc_feat + db_sc_feat

            r_f = self.deform_convs[idx](r_f)
            g_f = self.deform_convs[idx](g_f)
            b_f = self.deform_convs[idx](b_f)
            a_f = self.deform_convs[idx](a_f)
            d_f = self.deform_convs[idx](d_f)

        return r_f, g_f, b_f, a_f, d_f

class FeatureEnhanceDINoCross(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceDINoCross, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = nn.ModuleList()
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(self.depthLayers):
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_cals.append(CAM_Calculate(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))
            self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            #self.deform_convs.append(DeformableConv2d(in_channels=self.in_channels,
            #                                        out_channels=self.out_channels,
            #                                        kernel_size=3,
            #                                        padding=1))

    def forward(self, r_f, g_f, b_f, a_f):

        for idx in range(self.depthLayers):

            r_sp_feat = self.pams[idx](r_f)
            g_sp_feat = self.pams[idx](g_f)
            b_sp_feat = self.pams[idx](b_f)
            a_sp_feat = self.pams[idx](a_f)

            r_attention = self.cam_cals[idx](r_f)
            g_attention = self.cam_cals[idx](g_f)
            b_attention = self.cam_cals[idx](b_f)
            a_attention = self.cam_cals[idx](a_f)

            rr_sc_feat = self.cam_uses[idx](r_f, r_attention)
            gg_sc_feat = self.cam_uses[idx](g_f, g_attention)
            bb_sc_feat = self.cam_uses[idx](b_f, b_attention)
            aa_sc_feat = self.cam_uses[idx](a_f, a_attention)

            r_f = r_sp_feat + rr_sc_feat
            g_f = g_sp_feat + gg_sc_feat
            b_f = b_sp_feat + bb_sc_feat
            a_f = a_sp_feat + aa_sc_feat

            r_f = self.deform_convs[idx](r_f)
            g_f = self.deform_convs[idx](g_f)
            b_f = self.deform_convs[idx](b_f)
            a_f = self.deform_convs[idx](a_f)

        return r_f, g_f, b_f, a_f

class FeatureEnhanceDI(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceDI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = nn.ModuleList()
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(self.depthLayers):
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_cals.append(CAM_Calculate(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))
            self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            #self.deform_convs.append(DeformableConv2d(in_channels=self.in_channels,
            #                                        out_channels=self.out_channels,
            #                                        kernel_size=3,
            #                                        padding=1))

    def forward(self, r_f, g_f, b_f, a_f):

        for idx in range(self.depthLayers):

            r_sp_feat = self.pams[idx](r_f)
            g_sp_feat = self.pams[idx](g_f)
            b_sp_feat = self.pams[idx](b_f)
            a_sp_feat = self.pams[idx](a_f)

            r_attention = self.cam_cals[idx](r_f)
            g_attention = self.cam_cals[idx](g_f)
            b_attention = self.cam_cals[idx](b_f)
            a_attention = self.cam_cals[idx](a_f)

            rr_sc_feat = self.cam_uses[idx](r_f, r_attention)
            gg_sc_feat = self.cam_uses[idx](g_f, g_attention)
            bb_sc_feat = self.cam_uses[idx](b_f, b_attention)
            aa_sc_feat = self.cam_uses[idx](a_f, a_attention)

            rg_sc_feat = self.cam_uses[idx](r_f, g_attention)
            rb_sc_feat = self.cam_uses[idx](r_f, b_attention)
            ra_sc_feat = self.cam_uses[idx](r_f, a_attention)

            gr_sc_feat = self.cam_uses[idx](g_f, r_attention)
            gb_sc_feat = self.cam_uses[idx](g_f, b_attention)
            ga_sc_feat = self.cam_uses[idx](g_f, a_attention)

            bg_sc_feat = self.cam_uses[idx](b_f, g_attention)
            br_sc_feat = self.cam_uses[idx](b_f, r_attention)
            ba_sc_feat = self.cam_uses[idx](b_f, a_attention)

            ag_sc_feat = self.cam_uses[idx](a_f, g_attention)
            ab_sc_feat = self.cam_uses[idx](a_f, b_attention)
            ar_sc_feat = self.cam_uses[idx](a_f, r_attention)

            r_f = r_sp_feat + rr_sc_feat + rg_sc_feat + rb_sc_feat + ra_sc_feat
            g_f = g_sp_feat + gg_sc_feat + gr_sc_feat + gb_sc_feat + ga_sc_feat
            b_f = b_sp_feat + bb_sc_feat + br_sc_feat + bg_sc_feat + ba_sc_feat
            a_f = a_sp_feat + aa_sc_feat + ag_sc_feat + ab_sc_feat + ar_sc_feat

            r_f = self.deform_convs[idx](r_f)
            g_f = self.deform_convs[idx](g_f)
            b_f = self.deform_convs[idx](b_f)
            a_f = self.deform_convs[idx](a_f)

        return r_f, g_f, b_f, a_f

class FeatureEnhanceDepth(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceDepth, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cam_uses = nn.ModuleList()
        self.pams = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(self.depthLayers):
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))
            self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            #self.deform_convs.append(DeformableConv2d(in_channels=self.in_channels,
            #                                        out_channels=self.out_channels,
            #                                        kernel_size=3,
            #                                        padding=1))

    def forward(self, d_f):

        for idx in range(self.depthLayers):

            d_sp_feat = self.pams[idx](d_f)
            d_attention = self.cam_cals[idx](d_f)
            dd_sc_feat = self.cam_uses[idx](d_f, d_attention)
            d_f = d_sp_feat + dd_sc_feat
            d_f = self.deform_convs[idx](d_f)

        return d_f
    
class FeatureEnhanceRGB(nn.Module):

    depthLayers = 1

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceRGB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = nn.ModuleList()
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(self.depthLayers):
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_cals.append(CAM_Calculate(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))
            self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            #self.deform_convs.append(DeformableConv2d(in_channels=self.in_channels,
            #                                        out_channels=self.out_channels,
            #                                        kernel_size=3,
            #                                        padding=1))

    def forward(self, r_f, g_f, b_f):        

        for idx in range(self.depthLayers):

            r_sp_feat = self.pams[idx](r_f)
            g_sp_feat = self.pams[idx](g_f)
            b_sp_feat = self.pams[idx](b_f)

            r_attention = self.cam_cals[idx](r_f)
            g_attention = self.cam_cals[idx](g_f)
            b_attention = self.cam_cals[idx](b_f)

            rr_sc_feat = self.cam_uses[idx](r_f, r_attention)
            gg_sc_feat = self.cam_uses[idx](g_f, g_attention)
            bb_sc_feat = self.cam_uses[idx](b_f, b_attention)

            rg_sc_feat = self.cam_uses[idx](r_f, g_attention)
            rb_sc_feat = self.cam_uses[idx](r_f, b_attention)

            gr_sc_feat = self.cam_uses[idx](g_f, r_attention)
            gb_sc_feat = self.cam_uses[idx](g_f, b_attention)

            bg_sc_feat = self.cam_uses[idx](b_f, g_attention)
            br_sc_feat = self.cam_uses[idx](b_f, r_attention)

            r_f = r_sp_feat + rr_sc_feat + rg_sc_feat + rb_sc_feat
            g_f = g_sp_feat + gg_sc_feat + gr_sc_feat + gb_sc_feat
            b_f = b_sp_feat + bb_sc_feat + br_sc_feat + bg_sc_feat            

            r_f = self.deform_convs[idx](r_f)
            g_f = self.deform_convs[idx](g_f)
            b_f = self.deform_convs[idx](b_f)

        return r_f, g_f, b_f
class FeatureEnhanceWindow(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhanceWindow, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = PAM_Module(self.in_channels)
        self.cam_cals = CAM_Calculate(self.in_channels)
        self.cam_uses = CAM_Use(self.in_channels)
        self.deform_convs = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, image):
        imSpFeat = self.pams(image)
        imageReshaped = image.reshape((image.shape[0],image.shape[1],4,-1))
        attMap = imSpFeat.clone()
        for i in range(0,imageReshaped.shape[-1],4):
            wdtWindow = i+4 if i+4 < imageReshaped.shape[-1] else imageReshaped.shape[-1]
            slicedWindow = imageReshaped[:,:,:,i:wdtWindow]            
            rAttention = self.cam_cals(slicedWindow)
            rr_sc_feat = self.cam_uses(imSpFeat, rAttention)
            attMap += rr_sc_feat
        return attMap
        '''
        r_sp_feat = self.pams(r_f)

        r_attention = self.cam_cals(r_f)
        g_attention = self.cam_cals(g_f)
        b_attention = self.cam_cals(b_f)

        rr_sc_feat = self.cam_uses(r_f, r_attention)
        rg_sc_feat = self.cam_uses(r_f, g_attention)
        rb_sc_feat = self.cam_uses(r_f, b_attention)

        r_f = r_sp_feat + rr_sc_feat + rg_sc_feat + rb_sc_feat

        r_f = self.deform_convs(r_f)


        return r_f, g_f, b_f
        '''
