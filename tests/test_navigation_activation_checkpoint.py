"""Real navigation forward, gradients and dropout RNG with recomputation."""
import copy
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig
from vlnce_baselines.models.R1Policy import ETP
from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import GlocalTextPathNavCMT


def test_navigation_recomputation_preserves_outputs_gradients_and_rng():
    config = PretrainedConfig.from_pretrained('bert_config/xlm-roberta-base')
    for key, value in dict(hidden_size=32, num_attention_heads=4, intermediate_size=64,
        vocab_size=64, type_vocab_size=2, max_action_steps=100,
        max_txt_task_embeddings=4, max_gmap_task_embeddings=3,
        rgb_encoder_type='rae_dinov2', image_feat_size=768, depth_feat_size=128,
        angle_feat_size=4, num_l_layers=1, num_pano_layers=0, num_x_layers=1,
        graph_sprels=True, fix_lang_embedding=False, fix_pano_embedding=False,
        update_lang_bert=True, use_lang2visn_attn=True, use_depth_embedding=True,
        obj_feat_size=0, pred_head_dropout_prob=0.2, hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2).items():
        setattr(config,key,value)
    model = GlocalTextPathNavCMT(config).train()
    models = [model, copy.deepcopy(model)]
    source_txt, source_img = torch.randn(2,5,32), torch.randn(2,4,32)
    pos, distances = torch.randn(2,4,7), torch.rand(2,4,4)
    results = []
    for checkpointed, net in zip([False,True], models):
        policy = SimpleNamespace(vln_bert=net, training=True, checkpoint_navigation=checkpointed)
        txt, img = source_txt.clone().requires_grad_(), source_img.clone().requires_grad_()
        torch.manual_seed(919)
        logits = []
        for step in range(2):
            output = ETP.forward(policy, mode='navigation', txt_embeds=txt,
                txt_masks=torch.ones(2,5,dtype=torch.bool), gmap_vp_ids=[['0','1','2','3']]*2,
                gmap_step_ids=torch.full((2,4),step,dtype=torch.long), gmap_img_fts=img,
                gmap_pos_fts=pos, gmap_masks=torch.ones(2,4,dtype=torch.bool),
                gmap_visited_masks=torch.zeros(2,4,dtype=torch.bool),
                gmap_pair_dists=distances, gmap_task_embeddings=torch.ones(2,4,dtype=torch.long))
            logits.append(output['global_logits'])
        sum(torch.nn.functional.cross_entropy(x,torch.tensor([1,2])) for x in logits).backward()
        results.append((logits, txt.grad, img.grad,
            {n:p.grad for n,p in net.named_parameters() if p.grad is not None},torch.get_rng_state()))
    for left,right in zip(results[0][:3],results[1][:3]):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert results[0][3].keys() == results[1][3].keys()
    for name in results[0][3]:
        torch.testing.assert_close(results[0][3][name],results[1][3][name],rtol=0,atol=0)
    assert torch.equal(results[0][4],results[1][4])
