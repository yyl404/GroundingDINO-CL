# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # prompt tuning (disabled by default)
        self.prompt_embeddings = None
        self.prompt_mode = "shared"
        self.prompt_token_map = None
        self.aux_prompt_embeddings = None
        self.aux_prompt_mode = "shared"
        self.aux_prompt_token_map = None

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss 

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def init_prompt_tuning(self, prompt_length=16, init_std=0.02, num_embeddings=None, mode="shared"):
        if mode not in {"shared", "class_independent"}:
            raise ValueError(f"Unsupported prompt mode: {mode}")
        if num_embeddings is None:
            num_embeddings = prompt_length
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive.")
        prompt = torch.zeros(num_embeddings, self.hidden_dim)
        nn.init.normal_(prompt, mean=0.0, std=init_std)
        self.prompt_embeddings = nn.Parameter(prompt)
        self.prompt_mode = mode

    def init_aux_prompt_tuning(self, prompt_length=16, init_std=0.02, num_embeddings=None, mode="shared"):
        if mode not in {"shared", "class_independent"}:
            raise ValueError(f"Unsupported aux prompt mode: {mode}")
        if num_embeddings is None:
            num_embeddings = prompt_length
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive.")
        prompt = torch.zeros(num_embeddings, self.hidden_dim)
        nn.init.normal_(prompt, mean=0.0, std=init_std)
        self.aux_prompt_embeddings = nn.Parameter(prompt)
        self.aux_prompt_mode = mode

    def set_prompt_token_map(self, token_to_prompt_idx: Dict[int, int], max_text_len: Optional[int] = None):
        if self.prompt_embeddings is None:
            raise RuntimeError("Prompt tuning is not initialized. Call init_prompt_tuning first.")
        if max_text_len is None:
            max_text_len = self.max_text_len
        token_map = torch.full((max_text_len,), -1, dtype=torch.long)
        max_prompt_idx = self.prompt_embeddings.shape[0] - 1
        for token_idx, prompt_idx in token_to_prompt_idx.items():
            if token_idx < 0 or token_idx >= max_text_len:
                continue
            if prompt_idx < 0 or prompt_idx > max_prompt_idx:
                continue
            token_map[token_idx] = int(prompt_idx)
        self.prompt_token_map = token_map

    def set_aux_prompt_token_map(self, token_to_prompt_idx: Dict[int, int], max_text_len: Optional[int] = None):
        if self.aux_prompt_embeddings is None:
            raise RuntimeError("Aux prompt tuning is not initialized. Call init_aux_prompt_tuning first.")
        if max_text_len is None:
            max_text_len = self.max_text_len
        token_map = torch.full((max_text_len,), -1, dtype=torch.long)
        max_prompt_idx = self.aux_prompt_embeddings.shape[0] - 1
        for token_idx, prompt_idx in token_to_prompt_idx.items():
            if token_idx < 0 or token_idx >= max_text_len:
                continue
            if prompt_idx < 0 or prompt_idx > max_prompt_idx:
                continue
            token_map[token_idx] = int(prompt_idx)
        self.aux_prompt_token_map = token_map

    def clear_prompt_token_map(self):
        self.prompt_token_map = None

    def clear_aux_prompt_token_map(self):
        self.aux_prompt_token_map = None

    def freeze_except_prompt(self):
        for _, parameter in self.named_parameters():
            parameter.requires_grad_(False)
        if self.prompt_embeddings is None:
            raise RuntimeError("Prompt tuning is not initialized. Call init_prompt_tuning first.")
        self.prompt_embeddings.requires_grad_(True)

    def freeze_except_selected_prompts(self, train_main_prompt: bool, train_aux_prompt: bool):
        for _, parameter in self.named_parameters():
            parameter.requires_grad_(False)
        if train_main_prompt:
            if self.prompt_embeddings is None:
                raise RuntimeError("Main prompt tuning is not initialized.")
            self.prompt_embeddings.requires_grad_(True)
        if train_aux_prompt:
            if self.aux_prompt_embeddings is None:
                raise RuntimeError("Aux prompt tuning is not initialized.")
            self.aux_prompt_embeddings.requires_grad_(True)
        if not train_main_prompt and not train_aux_prompt:
            raise RuntimeError("At least one prompt branch should be trainable.")

    def get_prompt_state_dict(self):
        if self.prompt_embeddings is None:
            raise RuntimeError("Prompt tuning is not initialized. Call init_prompt_tuning first.")
        state_dict = {
            "prompt_embeddings": self.prompt_embeddings.detach().cpu(),
            "prompt_mode": self.prompt_mode,
        }
        if self.prompt_token_map is not None:
            state_dict["prompt_token_map"] = self.prompt_token_map.detach().cpu()
        if self.aux_prompt_embeddings is not None:
            state_dict["aux_prompt_embeddings"] = self.aux_prompt_embeddings.detach().cpu()
            state_dict["aux_prompt_mode"] = self.aux_prompt_mode
        if self.aux_prompt_token_map is not None:
            state_dict["aux_prompt_token_map"] = self.aux_prompt_token_map.detach().cpu()
        return state_dict

    def load_prompt_state_dict(self, state_dict):
        if "prompt_state_dict" in state_dict and isinstance(state_dict["prompt_state_dict"], dict):
            state_dict = state_dict["prompt_state_dict"]

        prompt = state_dict.get("prompt_embeddings")
        if prompt is None:
            raise KeyError("prompt_embeddings was not found in prompt state dict.")
        prompt = prompt.float()
        if self.prompt_embeddings is None or self.prompt_embeddings.shape != prompt.shape:
            self.prompt_embeddings = nn.Parameter(prompt.clone())
        else:
            with torch.no_grad():
                self.prompt_embeddings.copy_(prompt)

        self.prompt_mode = state_dict.get("prompt_mode", "shared")
        prompt_token_map = state_dict.get("prompt_token_map")
        if prompt_token_map is not None:
            self.prompt_token_map = prompt_token_map.long().clone()
        else:
            self.prompt_token_map = None

        aux_prompt = state_dict.get("aux_prompt_embeddings")
        if aux_prompt is not None:
            aux_prompt = aux_prompt.float()
            if self.aux_prompt_embeddings is None or self.aux_prompt_embeddings.shape != aux_prompt.shape:
                self.aux_prompt_embeddings = nn.Parameter(aux_prompt.clone())
            else:
                with torch.no_grad():
                    self.aux_prompt_embeddings.copy_(aux_prompt)
            self.aux_prompt_mode = state_dict.get("aux_prompt_mode", "shared")
            aux_prompt_token_map = state_dict.get("aux_prompt_token_map")
            if aux_prompt_token_map is not None:
                self.aux_prompt_token_map = aux_prompt_token_map.long().clone()
            else:
                self.aux_prompt_token_map = None
        else:
            self.aux_prompt_embeddings = None
            self.aux_prompt_mode = "shared"
            self.aux_prompt_token_map = None

    def _apply_prompt_embeddings(
        self,
        encoded_text: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor],
        prompt_token_map: Optional[torch.Tensor],
        prompt_token_map_override=None,
    ) -> torch.Tensor:
        if prompt_embeddings is None:
            return encoded_text

        token_map_to_use = prompt_token_map_override
        if token_map_to_use is None:
            token_map_to_use = prompt_token_map

        if token_map_to_use is None:
            prompt_len = min(encoded_text.shape[1], prompt_embeddings.shape[0])
            encoded_text[:, :prompt_len, :] = (
                encoded_text[:, :prompt_len, :] + prompt_embeddings[:prompt_len].unsqueeze(0)
            )
            return encoded_text

        if isinstance(token_map_to_use, list):
            if len(token_map_to_use) != encoded_text.shape[0]:
                raise ValueError("Length of prompt_token_map_override list must match batch size.")
            for batch_idx, token_map in enumerate(token_map_to_use):
                if token_map is None:
                    prompt_len = min(encoded_text.shape[1], prompt_embeddings.shape[0])
                    encoded_text[batch_idx : batch_idx + 1, :prompt_len, :] = (
                        encoded_text[batch_idx : batch_idx + 1, :prompt_len, :]
                        + prompt_embeddings[:prompt_len].unsqueeze(0)
                    )
                    continue
                valid_len = min(encoded_text.shape[1], token_map.shape[0])
                local_map = token_map[:valid_len].to(encoded_text.device)
                valid_mask = (local_map >= 0) & (local_map < prompt_embeddings.shape[0])
                if valid_mask.any():
                    token_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                    prompt_indices = local_map[token_positions]
                    encoded_text[batch_idx, token_positions, :] = (
                        encoded_text[batch_idx, token_positions, :] + prompt_embeddings[prompt_indices]
                    )
            return encoded_text

        valid_len = min(encoded_text.shape[1], token_map_to_use.shape[0])
        token_map = token_map_to_use[:valid_len].to(encoded_text.device)
        valid_mask = (token_map >= 0) & (token_map < prompt_embeddings.shape[0])
        if valid_mask.any():
            token_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            prompt_indices = token_map[token_positions]
            encoded_text[:, token_positions, :] = (
                encoded_text[:, token_positions, :] + prompt_embeddings[prompt_indices].unsqueeze(0)
            )
        return encoded_text

    def _encode_text_branch(
        self,
        captions: List[str],
        device,
        prompt_embeddings: Optional[torch.Tensor],
        prompt_token_map: Optional[torch.Tensor],
        prompt_token_map_override=None,
    ):
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        (
            text_self_attention_masks,
            position_ids,
            _cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)
        encoded_text = self.feat_map(bert_output["last_hidden_state"])
        encoded_text = self._apply_prompt_embeddings(
            encoded_text=encoded_text,
            prompt_embeddings=prompt_embeddings,
            prompt_token_map=prompt_token_map,
            prompt_token_map_override=prompt_token_map_override,
        )
        text_token_mask = tokenized.attention_mask.bool()
        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

    def _concat_text_branches(self, main_text_dict: Dict[str, torch.Tensor], aux_text_dict: Dict[str, torch.Tensor]):
        encoded_text = torch.cat([main_text_dict["encoded_text"], aux_text_dict["encoded_text"]], dim=1)
        text_token_mask = torch.cat(
            [main_text_dict["text_token_mask"], aux_text_dict["text_token_mask"]], dim=1
        )
        position_ids = torch.cat([main_text_dict["position_ids"], aux_text_dict["position_ids"]], dim=1)

        bs, main_len, _ = main_text_dict["encoded_text"].shape
        aux_len = aux_text_dict["encoded_text"].shape[1]
        text_self_attention_masks = torch.zeros(
            (bs, main_len + aux_len, main_len + aux_len),
            dtype=main_text_dict["text_self_attention_masks"].dtype,
            device=main_text_dict["text_self_attention_masks"].device,
        )
        text_self_attention_masks[:, :main_len, :main_len] = main_text_dict["text_self_attention_masks"]
        text_self_attention_masks[:, main_len:, main_len:] = aux_text_dict["text_self_attention_masks"]

        if encoded_text.shape[1] > self.max_text_len:
            keep_main = min(main_len, self.max_text_len)
            keep_aux = min(aux_len, max(self.max_text_len - keep_main, 0))
            encoded_text = torch.cat(
                [encoded_text[:, :keep_main, :], encoded_text[:, main_len : main_len + keep_aux, :]], dim=1
            )
            text_token_mask = torch.cat(
                [
                    text_token_mask[:, :keep_main],
                    text_token_mask[:, main_len : main_len + keep_aux],
                ],
                dim=1,
            )
            position_ids = torch.cat(
                [position_ids[:, :keep_main], position_ids[:, main_len : main_len + keep_aux]], dim=1
            )
            text_self_attention_masks = torch.zeros(
                (bs, keep_main + keep_aux, keep_main + keep_aux),
                dtype=main_text_dict["text_self_attention_masks"].dtype,
                device=main_text_dict["text_self_attention_masks"].device,
            )
            text_self_attention_masks[:, :keep_main, :keep_main] = main_text_dict[
                "text_self_attention_masks"
            ][:, :keep_main, :keep_main]
            if keep_aux > 0:
                text_self_attention_masks[:, keep_main:, keep_main:] = aux_text_dict[
                    "text_self_attention_masks"
                ][:, :keep_aux, :keep_aux]

        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        main_prompt_token_map_override = kw.get("prompt_token_map_override")
        main_text_dict = self._encode_text_branch(
            captions=captions,
            device=samples.device,
            prompt_embeddings=self.prompt_embeddings,
            prompt_token_map=self.prompt_token_map,
            prompt_token_map_override=main_prompt_token_map_override,
        )
        aux_captions = kw.get("aux_captions")
        if aux_captions is not None:
            aux_prompt_token_map_override = kw.get("aux_prompt_token_map_override")
            aux_text_dict = self._encode_text_branch(
                captions=aux_captions,
                device=samples.device,
                prompt_embeddings=self.aux_prompt_embeddings,
                prompt_token_map=self.aux_prompt_token_map,
                prompt_token_map_override=aux_prompt_token_map_override,
            )
            text_dict = self._concat_text_branches(main_text_dict, aux_text_dict)
        else:
            text_dict = main_text_dict

        # import ipdb; ipdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(self.features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](self.features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, self.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor() ## If necessary
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    return model

