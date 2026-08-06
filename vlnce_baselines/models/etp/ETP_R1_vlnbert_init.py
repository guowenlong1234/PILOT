import torch


def _validate_pretrain_rgb_interface(state_dict, rgb_encoder_type):
    projection_marker = 'bert.img_embeddings.rgb_projection.'
    legacy_projection = sorted(
        key for key in state_dict if key.startswith(projection_marker)
    )
    if legacy_projection:
        raise ValueError(
            'The pretrained checkpoint uses the retired 768->512 '
            'rgb_projection pipeline and is incompatible with the '
            'ETPNav-compatible RGB interface: '
            + ', '.join(legacy_projection)
        )

    if rgb_encoder_type != 'rae_dinov2':
        return

    weight_key = 'bert.img_embeddings.img_linear.weight'
    bias_key = 'bert.img_embeddings.img_linear.bias'
    missing = [
        key for key in (weight_key, bias_key) if key not in state_dict
    ]
    if missing:
        raise ValueError(
            'RAE/DINOv2 pretrained checkpoint is missing the 768-dimensional '
            'ETPNav visual interface: ' + ', '.join(missing)
        )
    expected_shapes = {
        weight_key: (768, 768),
        bias_key: (768,),
    }
    for key, expected_shape in expected_shapes.items():
        value = state_dict[key]
        actual_shape = tuple(value.shape) if torch.is_tensor(value) else None
        if actual_shape != expected_shape:
            raise ValueError(
                f'RAE/DINOv2 pretrained checkpoint {key} must have shape '
                f'{expected_shape}, got {actual_shape}'
            )


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None, dropout_rate=0.1):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    keywords = ['graph_query_text', 'graph_attentioned_txt_embeds_transform', 'global_sap_head']
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        for original_key, value in ckpt_weights.items():
            normalized_key = (
                original_key[7:]
                if original_key.startswith('module.')
                else original_key
            )
            if (
                any(keyword in normalized_key for keyword in keywords)
                and not normalized_key.startswith('bert.')
            ):
                normalized_key = 'bert.' + normalized_key
            new_ckpt_weights[normalized_key] = value

    rgb_encoder_type = str(config.RGB_ENCODER.type).lower()
    _validate_pretrain_rgb_interface(new_ckpt_weights, rgb_encoder_type)
    
    cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.rgb_encoder_type = config.RGB_ENCODER.type
    vis_config.image_feat_size = config.RGB_ENCODER.output_size
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 12
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True

    vis_config.pred_head_dropout_prob = dropout_rate
    vis_config.hidden_dropout_prob = dropout_rate
    vis_config.attention_probs_dropout_prob = dropout_rate

    vis_config.use_lang2visn_attn = True

    vis_config.max_txt_task_embeddings = 4
    vis_config.max_gmap_task_embeddings = 3

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    return visual_model
