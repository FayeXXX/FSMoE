from utils.cocoop_utils import str2bool
import torch.nn as nn
from torch.nn import functional as F
import argparse
from collections import OrderedDict
import warnings
import pickle
import torch
import os.path as osp
from functools import partial

from clip_revised import clip
from clip_revised.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image

# import Swin Transformer
from models.cat import CAT
_tokenizer = _Tokenizer()


class LayerNorm(nn.LayerNorm):
    def forward(self, x:torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        fpath = 'log/my_model/model.pth.tar-10'
        checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
     weight_path = 'log/my_model/model-best.pth.tar'
     load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )


def load_clip_to_cpu(args):
    backbone_name = args.backbone
    # model_path = clip._download(url)

    model_path = f"/home/xyf/PretrainedModels/CLIP/{backbone_name}.pt"
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoCoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class FeatureFusionNetwork(nn.Module):
    def __init__(self, input_dim_shallow, input_dim_final, output_dim):
        super(FeatureFusionNetwork, self).__init__()
        self.input_dim_shallow = input_dim_shallow
        self.input_dim_final = input_dim_final
        self.output_dim = output_dim
        # 将浅层特征映射到与最终层特征相同的维度
        self.q_proj = nn.Linear(input_dim_final, input_dim_final)
        self.k_proj = nn.Linear(input_dim_shallow, input_dim_final)
        self.v_proj = nn.Linear(input_dim_shallow, input_dim_final)
        # 定义用于输出的线性层
        self.linear1 = nn.Linear(input_dim_final, output_dim)

    def forward(self, im_features, lowfeature):
        transformed_feature_k = self.k_proj(lowfeature)  # [batch_size, num_shallow_layers, input_dim_final]
        transformed_feature_v = self.v_proj(lowfeature)

        expanded_final_feature = im_features.unsqueeze(1).expand(-1, len(lowfeature), -1)
        final_feature_q = self.q_proj(expanded_final_feature)  # [batch_size, 1, input_dim_final]

        attention_scores = (transformed_feature_k * final_feature_q).sum(dim=-1)  # [batch_size, num_layers]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_layers]
        weighted_sum_features = (transformed_feature_v * attention_weights.unsqueeze(-1)).sum(dim=1)
        weighted_sum_features = self.linear1(weighted_sum_features)

        return weighted_sum_features


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        ctx_init = cfg.CTX_INIT
        dtype = clip_model.dtype  # torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        vis_dim = clip_model.visual.output_dim   # 512
        trans_dim = clip_model.visual.ln_post.normalized_shape[0]   # 768
        clip_imsize = clip_model.visual.input_resolution   # 224
        cfg_imsize = cfg.image_size  # 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")   # "a photo of a"
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)  # (1, n_tkn)  the length of prompt tokens in CLIP is 77 (n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)  # (1, 77, 512)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # (4, 512)
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),   # (512, 32)
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))   # (32, 512)
        ]))

        width = clip_model.visual.transformer.width
        self.proj_net = nn.Sequential(OrderedDict([
            ("layernorm", LayerNorm(width)),
            ("linear", nn.Linear(width, ctx_dim))
        ]))
        nn.init.normal_(self.proj_net.linear.weight, mean=0.0, std=width ** -0.5)

        # self.experts = cfg.layers
        # if self.experts:
        #     self.expert_embeddings = nn.Linear(ctx_dim, len(self.experts))
        #     self.softmax = nn.Softmax(dim=1)

        if cfg.prec == "fp16":
            self.meta_net.half()
            self.proj_net.half()
            self.expert_embeddings.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn) one-hot (77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)    # (n_cls, n_tkn, 512)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS  (n_cls, 1, 512)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS (n_cls, n_tkn-1-n_ctx, 512)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        # self.extra_weight = cfg.extra_weight

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts


    def forward(self, im_features):
        top_feature = self.meta_net(im_features[-1])
        if len(im_features) > 1:
            other_features = self.proj_net(torch.stack(im_features[:-1]))

            #weights = self.softmax(self.expert_embeddings(top_feature))      # MOE weights
            #bias = top_feature + self.extra_weight * (weights.unsqueeze(2) * other_features.transpose(0, 1)).sum(dim=1)
            bias = torch.cat([top_feature.unsqueeze(1), other_features.transpose(0, 1)], dim=1)
            bias = bias.mean(dim=1)
        else:
            bias = top_feature
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label):   # (batch_size, channels, img_size, img_size)
        tokenized_prompts = self.tokenized_prompts   # (n_cls, n_tkn)
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))        # 提取各transformer层的特征
        image_features[-1] = image_features[-1] / image_features[-1].norm(dim=-1, keepdim=True)   # (batch_size, 512)

        selected_features = [image_features[i] for i in self.cfg.layers] if self.cfg.layers else []
        selected_features.append(image_features[-1])

        prompts = self.prompt_learner(selected_features)  # (batch_size, n_cls, n_tkn, ctx_dim)
        logits = []
        for pts_i, imf_i in zip(prompts, image_features[-1]):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def get_cocoop_model(args, log):
     # Get model
    clip_model = load_clip_to_cpu(args)
    # log.info("Building custom CLIP")

    if args.prec == "fp32" or args.prec == "amp":
         # CLIP's default precision is fp16
        clip_model.float()

    model = CustomCLIP(args, args.target_classes, clip_model)
    # total_params = sum(p.numel() for p in model.parameters())
    # log.info("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            param.requires_grad_(False)

     # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    # print(f"Parameters to be updated: {enabled}")

    if args.INIT_WEIGHTS:
        load_pretrained_weights(model.prompt_learner, args.INIT_WEIGHTS)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--model', default='timm_resnet50_pretrained', type=str)
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    parser.add_argument('--loss', type=str, default='ARPLoss')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--backbone', type=str, default='ViT-B-16')
    parser.add_argument('--INIT_WEIGHTS', default=False, type=str2bool)
    parser.add_argument('--N_CTX', type=int, default=4)
    parser.add_argument('--CTX_INIT', type=str, default="a photo of a")
    parser.add_argument('--prec', type=str, default='fp32')
    parser.add_argument('--oriweight', type=float, default=1.0, help="weight for features of original images")
    parser.add_argument('--clipweight', type=float, default=1.0, help="weight for features of CLIP images")
    parser.add_argument('--alphaweight', type=float, default=0.6, help="weight for shallow features")
    parser.add_argument('--betaweight', type=float, default=0.4, help="weight for last layer features")
    parser.add_argument('--layers', type=int, nargs='*', default=[0,1,2])
    parser.add_argument('--extra_weight', type=float, default=0.3, help='Path to the file containing layer indices.')

    args = parser.parse_args()

    args.target_classes = ['DH-82','DHC-1','DHC-6']
    log = None
    model = get_cocoop_model(args, log)

    if args.prec == "fp16":
        debug_input = torch.randn(64, 3, 32, 32, dtype=torch.float16)
    else:
        debug_input = torch.randn(64, 3, 32, 32, dtype=torch.float32)

    # debug_input = debug_input.float()

    transform = Compose([
        ToPILImage(),
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
    ])

    debug_input = torch.stack([transform(image) for image in debug_input])

    x, y = model(debug_input, None)
    debug = True