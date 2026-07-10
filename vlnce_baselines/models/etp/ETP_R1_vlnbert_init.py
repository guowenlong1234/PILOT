import torch


_RGB_PROJECTION_PARAMETER_NAMES = (
    '0.weight',
    '0.bias',
    '2.weight',
    '2.bias',
    '4.weight',
    '4.bias',
)


def _validate_pretrain_rgb_projection(state_dict, rgb_encoder_type):
    prefix = 'bert.img_embeddings.rgb_projection.'
    expected = {prefix + name for name in _RGB_PROJECTION_PARAMETER_NAMES}
    present = {key for key in state_dict if key.startswith(prefix)}

    if rgb_encoder_type == 'rae_dinov2':
        missing = sorted(expected - present)
        if missing:
            raise ValueError(
                'RAE/DINOv2 pretrained checkpoint requires a complete '
                'rgb_projection; missing required keys: '
                + ', '.join(missing)
            )
    elif present:
        raise ValueError(
            'CLIP cannot load a RAE/DINOv2 pretrained checkpoint containing '
            + ', '.join(sorted(present))
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
    _validate_pretrain_rgb_projection(new_ckpt_weights, rgb_encoder_type)
    
    cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.rgb_encoder_type = config.RGB_ENCODER.type
    vis_config.raw_image_feat_size = config.RGB_ENCODER.raw_output_size
    vis_config.image_feat_size = config.RGB_ENCODER.output_size
    vis_config.projection_hidden_size = config.RGB_ENCODER.projection_hidden_size
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
