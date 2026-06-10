import warnings
from typing import Dict, List, Sequence, Tuple, Union
import math
import re

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.models.GroundingDINO.utils import MLP
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank `r` must be positive, got {r}.")
        self.base = base
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def segment_text_to_word_embeddings(
    text: str,
    tokenizer,
    bert,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    batch = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = batch["input_ids"].to(device)
    wemb = bert.embeddings.word_embeddings(ids).squeeze(0).to(dtype=dtype)
    return wemb, ids.squeeze(0)


class GroundingDINOWrapper(nn.Module):
    def __init__(
        self,
        model: GroundingDINO,
        classes: List[str],
        prompt_len=4,
        text_mode: str = "prompt",
        inject_before_encoder=True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Sequence[str] = ("value_proj", "output_proj", "linear1", "linear2"),
        lora_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if text_mode not in ("prompt", "fixed"):
            raise ValueError(f"text_mode must be 'prompt' or 'fixed', got {text_mode!r}.")
        self.model = model
        self.classes = classes
        self.prompt_len = prompt_len
        self.text_mode = text_mode
        self.inject_before_encoder = inject_before_encoder
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = tuple(lora_targets)
        self.lora_layers = None if lora_layers is None else tuple(sorted({int(i) for i in lora_layers}))

        if self.text_mode == "prompt":
            self._init_embeddings()
        else:
            self.embeddings = None

        hidden_dim = self.model.hidden_dim
        self.bbox_embed_last_layer = MLP(hidden_dim, hidden_dim, 4, 3)
        self._zero_init_module(self.bbox_embed_last_layer) # zero initial to maintain zero-shot performance

        num_classes = len(classes)
        self.cls_head = self._build_cls_head(num_classes)
        self._zero_init_module(self.cls_head)

        if self.use_lora:
            self._inject_lora_into_model()

    def _build_cls_head(self, num_classes: int, device=None, dtype=None):
        hidden_dim = self.model.hidden_dim
        head = nn.Sequential(
            # MLP(hidden_dim, hidden_dim, hidden_dim, 2),
            nn.Linear(hidden_dim, num_classes, bias=False),
        )
        if device is not None:
            head.to(device)
        if dtype is not None:
            head.to(dtype)
        return head

    @staticmethod
    def _zero_init_module(module: nn.Module) -> None:
        for p in module.parameters():
            nn.init.zeros_(p)

    def _inject_lora_into_model(self) -> None:
        if not self.lora_targets:
            raise ValueError("`lora_targets` cannot be empty when use_lora=True.")

        replaced_count = 0
        module_names = [name for name, _ in self.model.named_modules()]
        for module_name in module_names:
            if not module_name:
                continue
            parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            child = getattr(parent, child_name)
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            if not any(target in module_name for target in self.lora_targets):
                continue
            if not self._is_in_selected_lora_layers(module_name):
                continue
            setattr(
                parent,
                child_name,
                LoRALinear(
                    child,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                ),
            )
            replaced_count += 1

        if replaced_count == 0:
            warnings.warn(
                "LoRA is enabled but no Linear layers matched "
                f"targets={self.lora_targets}, layers={self.lora_layers}.",
                UserWarning,
                stacklevel=2,
            )

    def _is_in_selected_lora_layers(self, module_name: str) -> bool:
        if self.lora_layers is None:
            return True

        patterns = (
            r"transformer\.encoder\.layers\.(\d+)\.",
            r"transformer\.encoder\.text_layers\.(\d+)\.",
            r"transformer\.decoder\.layers\.(\d+)\.",
        )
        for pattern in patterns:
            m = re.search(pattern, module_name)
            if m:
                return int(m.group(1)) in self.lora_layers
        return False

    def _init_embeddings(self) -> None:
        prompt_len = self.prompt_len
        num_classes = len(self.classes)
        encoder_dim = self.model.bert.config.hidden_size
        if num_classes == 0:
            self.embeddings = nn.Parameter(torch.empty(0, prompt_len, encoder_dim))
            return
        tokenizer = self.model.tokenizer
        bert = self.model.bert
        param = next(bert.parameters())
        device, dtype = param.device, param.dtype
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        pad_ids = torch.tensor([[pad_id]], device=device, dtype=torch.long)
        pad_wemb = bert.embeddings.word_embeddings(pad_ids).squeeze(0).to(dtype=dtype)
        embeddings = pad_wemb.expand(num_classes, prompt_len, encoder_dim).clone()

        with torch.no_grad():   
            if self.inject_before_encoder:
                for i, name in enumerate(self.classes):
                    tokenized = tokenizer(name, add_special_tokens=False, return_tensors="pt")
                    input_ids = tokenized["input_ids"].to(device)
                    seq_len = int(input_ids.shape[1])
                    n = min(seq_len, prompt_len)
                    if n == 0:
                        continue
                    tok_ids = input_ids[:, :n]
                    wemb = bert.embeddings.word_embeddings(tok_ids).squeeze(0)
                    embeddings[i, :n] = wemb.to(dtype=dtype)
            else:
                was_train = bert.training
                bert.eval()
                for i, name in enumerate(self.classes):
                    tokenized = tokenizer(name, add_special_tokens=False, return_tensors="pt")
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}
                    h = bert(**tokenized)["last_hidden_state"][0].to(dtype=dtype)
                    n = min(int(h.shape[0]), prompt_len)
                    if n > 0:
                        embeddings[i, :n] = h[:n]
                bert.train(was_train)
        
        self.embeddings = nn.Parameter(embeddings)

    @staticmethod
    def _broadcast_text_dict(text_dict: Dict[str, Tensor], batch_size: int) -> Dict[str, Tensor]:
        out = {}
        for key, value in text_dict.items():
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"text_dict['{key}'] must be a torch.Tensor, got {type(value)}"
                )
            ndim = value.dim()
            if ndim < 2:
                raise ValueError(
                    f"text_dict['{key}'] must have at least 2 dims [B, ...], "
                    f"got shape={tuple(value.shape)}"
                )
            if value.shape[0] == 1 and batch_size > 1:
                if ndim == 2:
                    out[key] = value.repeat(batch_size, 1)
                elif ndim == 3:
                    out[key] = value.repeat(batch_size, 1, 1)
                else:
                    raise ValueError(
                        f"Unsupported rank for broadcasting text_dict['{key}']: "
                        f"shape={tuple(value.shape)} (ndim={ndim}), only ndim=2/3 are supported"
                    )
            elif value.shape[0] == batch_size:
                out[key] = value
            else:
                raise ValueError(
                    f"text_dict['{key}'] batch mismatch: tensor batch={value.shape[0]}, "
                    f"image batch={batch_size}, shape={tuple(value.shape)}"
                )

        return out

    def _encode_fixed_text(self, classes: List[str], batch_size: int) -> Tuple[Dict[str, Tensor], Tensor]:
        model = self.model
        device = next(model.parameters()).device

        caption_text = ""
        token_spans = []
        for cls_name in classes:
            token_spans.append([[len(caption_text), len(caption_text) + len(cls_name)]])
            caption_text += (cls_name + ".")
        captions = [caption_text for _ in range(batch_size)]

        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption_text),
            token_span=token_spans,
        ).to(device)

        tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.specical_tokens, model.tokenizer
        )

        if text_self_attention_masks.shape[1] > model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : model.max_text_len, : model.max_text_len
            ]
            position_ids = position_ids[:, : model.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : model.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : model.max_text_len]

        if model.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = model.bert(**tokenized_for_encoder)
        encoded_text = model.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized["attention_mask"].bool()

        if encoded_text.shape[1] > model.max_text_len:
            encoded_text = encoded_text[:, : model.max_text_len, :]
            text_token_mask = text_token_mask[:, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : model.max_text_len, : model.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }
        return text_dict, positive_maps

    def concat_embeddings(self, classes: List[str]):
        model = self.model
        tokenizer = model.tokenizer
        bert = model.bert
        param = next(bert.parameters())
        device, dtype = param.device, param.dtype
        max_len = int(getattr(model, "max_text_len", 256))

        classes_ids = [self.classes.index(c) for c in classes]
        embeddings = self.embeddings[classes_ids]
        num_classes, prompt_len, embed_dim = embeddings.shape

        if self.inject_before_encoder:
            seperator_embeddings, _ = segment_text_to_word_embeddings(
                ".", tokenizer, bert, device, dtype
            )
            cls_embeddings, _ = segment_text_to_word_embeddings(
                tokenizer.cls_token, tokenizer, bert, device, dtype
            )
            sep_embeddings, _ = segment_text_to_word_embeddings(
                tokenizer.sep_token, tokenizer, bert, device, dtype
            )

            inputs_embeds = []
            inputs_embeds.append(cls_embeddings)
            for emb in embeddings:
                inputs_embeds.extend([emb, seperator_embeddings])
            inputs_embeds.append(sep_embeddings)
            inputs_embeds = torch.concat(inputs_embeds, dim=0).unsqueeze(0)
            seq_len = inputs_embeds.shape[1]
            n = min(max_len, seq_len)
            token_type_ids = torch.zeros(1, seq_len, device=device, dtype=torch.long)

            sub_sentence_ids = [0]
            for i in range(num_classes):
                sub_sentence_ids.extend([i + 1] * (prompt_len + 1))
            sub_sentence_ids.append(num_classes + 1)
            sub_sentence_ids = torch.tensor(
                sub_sentence_ids, device=device, dtype=torch.long
            )
            text_self_attention_masks = (
                sub_sentence_ids[None, :] == sub_sentence_ids[:, None]
            ).unsqueeze(0)
            
            position_ids = [0]
            for _ in range(num_classes):
                position_ids.extend(list(range(prompt_len + 1)))
            position_ids.append(0)
            position_ids = torch.tensor(
                position_ids, device=device, dtype=torch.long
            ).unsqueeze(0)

            text_self_attention_masks = text_self_attention_masks[:, :n, :n]
            position_ids = position_ids[:, :n]

            tokenized = {
                "inputs_embeds": inputs_embeds[:, :n],
                "token_type_ids": token_type_ids[:, :n],
                "attention_mask": torch.ones(1, n, device=device, dtype=torch.long),
            }

            positive_maps = torch.zeros(num_classes, max_len, device=device, dtype=torch.float32)
            for i in range(num_classes):
                beg = 1 + i * (prompt_len + 1)
                positive_maps[i, beg : beg + prompt_len] = 1.0
            positive_maps = positive_maps / (positive_maps.sum(dim=-1, keepdim=True) + 1e-6)

            return tokenized, text_self_attention_masks, position_ids, positive_maps
        else:
            encoded_text = torch.reshape(embeddings, [1, num_classes * prompt_len, embed_dim])
            # Keep text width consistent with transformer d_model (e.g. 256).
            encoded_text = model.feat_map(encoded_text)
            text_token_mask = torch.ones([1, num_classes * prompt_len], device=device, dtype=torch.bool)
            
            position_ids = []
            for _ in range(num_classes):
                position_ids.extend(list(range(prompt_len)))
            position_ids = torch.tensor(
                position_ids, device=device, dtype=torch.long
            ).unsqueeze(0)
            
            sub_sentence_ids = []
            for i in range(num_classes):
                sub_sentence_ids.extend([i] * (prompt_len))
            sub_sentence_ids = torch.tensor(
                sub_sentence_ids, device=device, dtype=torch.long
            )
            text_self_attention_masks = (
                sub_sentence_ids[None, :] == sub_sentence_ids[:, None]
            ).unsqueeze(0)

            n = min(max_len, num_classes*prompt_len)
            text_dict = {
                "encoded_text": encoded_text[:, :n],  # bs, 195, d_model
                "text_token_mask": text_token_mask[:, :n],  # bs, 195
                "position_ids": position_ids[:, :n],  # bs, 195
                "text_self_attention_masks": text_self_attention_masks[:, :n, :n],  # bs, 195,195
            }

            positive_maps = torch.zeros(num_classes, max_len, device=device, dtype=torch.float32)
            for i in range(num_classes):
                beg = i * (prompt_len)
                positive_maps[i, beg : beg + prompt_len] = 1.0
            positive_maps = positive_maps / (positive_maps.sum(dim=-1, keepdim=True) + 1e-6)

            return text_dict, positive_maps

    def _aggregate_class_logits(
        self,
        logits: Tensor,
        positive_maps: Tensor,
        aggregation_method: str,
    ) -> Tensor:
        if aggregation_method == "mean":
            # positive_maps is normalized when created
            class_logits = torch.einsum("ct,bqt->bqc", positive_maps, logits)
        elif aggregation_method == "sum":
            positive_maps_unnormalized = (positive_maps > 1e-6).to(dtype=logits.dtype)
            class_logits = torch.einsum("ct,bqt->bqc", positive_maps_unnormalized, logits)
            class_logits = torch.clamp(class_logits, 0.0, 1.0)
        elif aggregation_method == "max":
            token_mask = positive_maps > 1e-6
            class_logits = logits[:, None, :, :].masked_fill(~token_mask[None, :, None, :], float("-inf"))
            class_logits = class_logits.max(dim=-1).values.transpose(1, 2)
        elif aggregation_method == "min":
            token_mask = positive_maps > 1e-6
            class_logits = logits[:, None, :, :].masked_fill(~token_mask[None, :, None, :], float("inf"))
            class_logits = class_logits.min(dim=-1).values.transpose(1, 2)
        else:
            raise ValueError(f"Aggregation method {aggregation_method} is not available. Current choice: 'mean', 'sum', 'max', 'min'")
        return class_logits

    def forward(self, samples: Union[NestedTensor, List[Tensor]], classes:List[str]=None, **kw):
        # If the target classes are not designated, use the built-in vocabulary
        if classes is None:
            classes = self.classes
        else:
            unknown = [_c for _c in classes if _c not in self.classes]
            if unknown: warnings.warn(f"Ignoring classes not in self.classes: {unknown}", UserWarning, stacklevel=2)
            classes = [_c for _c in classes if _c in self.classes]
        
        model = self.model

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        batch_size = int(samples.tensors.shape[0])

        if self.text_mode == "fixed":
            text_dict, positive_maps = self._encode_fixed_text(classes, batch_size)
        elif self.inject_before_encoder:
            tokenized, text_self_attention_masks, position_ids, positive_maps = self.concat_embeddings(classes)

            if model.sub_sentence_present:
                tokenized_for_encoder = {
                    k: v for k, v in tokenized.items() if k != "attention_mask"
                }
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                tokenized_for_encoder = tokenized

            bert_output = model.bert(**tokenized_for_encoder)  # bs, 195, 768

            encoded_text = model.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            text_token_mask = tokenized["attention_mask"].bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > model.max_text_len:
                encoded_text = encoded_text[:, : model.max_text_len, :]
                text_token_mask = text_token_mask[:, : model.max_text_len]
                position_ids = position_ids[:, : model.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : model.max_text_len, : model.max_text_len
                ]

            text_dict = {
                "encoded_text": encoded_text,  # bs, 195, d_model
                "text_token_mask": text_token_mask,  # bs, 195
                "position_ids": position_ids,  # bs, 195
                "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
            }
        else:
            text_dict, positive_maps = self.concat_embeddings(classes)

        # import ipdb; ipdb.set_trace()
        text_dict = self._broadcast_text_dict(text_dict, batch_size)
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            model.set_image_tensor(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(model.features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](model.features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                model.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(
            srcs, masks, input_query_bbox, model.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], model.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            if dec_lid == len(hs) - 1:  # apply extra bbox refinement on the final decoder layer
                layer_delta_unsig += self.bbox_embed_last_layer(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        pred_boxes = outputs_coord_list[-1]

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(model.class_embed, hs)
            ]
        )
        delta_class_logits_unsig = self.cls_head(hs[-1])
        cid = torch.tensor([self.classes.index(c) for c in classes],
                           device=delta_class_logits_unsig.device,
                           dtype=torch.long)
        delta_class_logits_unsig = delta_class_logits_unsig.index_select(-1, cid)
        out = {"pred_logits": outputs_class[-1], "pred_boxes": pred_boxes}

        # # for intermediate outputs
        # if model.aux_loss:
        #     out['aux_outputs'] = model._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = model.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            model.unset_image_tensor() ## If necessary

        # aggregate token-wise logits to class-wise logits
        aggregation_method = kw.get("aggregation_method", "max")
        logits = outputs_class[-1].sigmoid()  # bs, nq, ntoken
        class_logits = self._aggregate_class_logits(logits, positive_maps, aggregation_method)
        
        class_logits = (inverse_sigmoid(class_logits) + delta_class_logits_unsig).sigmoid()
        out["pred_class_logits"] = class_logits

        return out

    def decode_embeddings(self, classes):
        # If the target classes are not designated, use the built-in vocabulary
        if classes is None:
            classes = self.classes
        # Filter invalid classes
        else:
            unknown = [_c for _c in classes if _c not in self.classes]
            if unknown: warnings.warn(f"Ignoring classes not in self.classes: {unknown}", UserWarning, stacklevel=2)
            classes = [_c for _c in classes if _c in self.classes]
        
        if self.text_mode != "prompt":
            raise NotImplementedError(
                "decode_embeddings is only supported when text_mode='prompt'."
            )

        if self.inject_before_encoder:
            if len(classes) == 0:
                return {}

            model = self.model
            tokenizer = model.tokenizer
            bert = model.bert

            classes_ids = [self.classes.index(c) for c in classes]
            embeddings = self.embeddings[classes_ids]  # [num_classes, prompt_len, hidden]
            num_classes, prompt_len, _ = embeddings.shape

            # Find nearest token in the BERT word embedding table for each prompt vector.
            word_embeddings = bert.embeddings.word_embeddings.weight  # [vocab_size, hidden]
            class_vectors = embeddings.reshape(num_classes * prompt_len, -1)

            class_vectors = F.normalize(class_vectors, p=2, dim=-1)
            word_embeddings = F.normalize(word_embeddings, p=2, dim=-1)
            similarity = torch.matmul(class_vectors, word_embeddings.t())  # [num_classes*prompt_len, vocab_size]
            nearest_scores, nearest_token_ids = similarity.max(dim=-1)
            nearest_token_ids = nearest_token_ids.reshape(num_classes, prompt_len)
            nearest_scores = nearest_scores.reshape(num_classes, prompt_len)

            decoded = {}
            for cls_name, token_ids, token_scores in zip(classes, nearest_token_ids, nearest_scores):
                token_id_list = token_ids.detach().cpu().tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_id_list)
                decoded_text = tokenizer.convert_tokens_to_string(tokens).strip()
                decoded[cls_name] = {
                    "decoded_text": decoded_text,
                    "decoded_tokens": tokens,
                    "cosine_similarities": token_scores.detach().cpu().tolist(),
                }

            return decoded
        else:
            raise NotImplementedError(
                "decode_embeddings is only supported when inject_before_encoder=True; "
                "current mode (inject_before_encoder=False) uses post-encoder features "
                "that cannot be directly mapped back to tokenizer word embeddings."
            )
