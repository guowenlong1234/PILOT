#!/usr/bin/env python3
"""Decode captured NWM patch tokens with the read-only original RAE decoder."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    upstream = Path(args.upstream).resolve()
    root = Path(args.capture).resolve()
    out = root.parent / 'visuals'
    out.mkdir(exist_ok=True)
    # Import just the standalone decoder package, not the upstream encoder or runtime.
    sys.path.insert(0, str(upstream / 'RAE/src/stage1'))
    decoder_module = importlib.import_module('decoders')
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(upstream / 'RAE/configs/decoder/ViTXL'))
    config.hidden_size, config.patch_size, config.image_size = 768, 16, 256
    decoder = decoder_module.GeneralDecoder(config, num_patches=256)
    state = torch.load(upstream / 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt', map_location='cpu')
    decoder.load_state_dict(state, strict=True)
    decoder.requires_grad_(False).eval().to(args.device)
    processor = json.loads((upstream / 'models/encoders/dinov2-with-registers-base/preprocessor_config.json').read_text())
    image_mean = torch.tensor(processor['image_mean'], device=args.device).view(1,3,1,1)
    image_std = torch.tensor(processor['image_std'], device=args.device).view(1,3,1,1)
    records = []
    thumbs = []
    for path in sorted(root.glob('sample_*.pt')):
        item = torch.load(path, map_location='cpu', weights_only=False)
        meta = item['meta']
        # Identical decoder applied to source, target truth, and prediction.
        tokens = torch.stack([item['truth_tokens'][-2,1:], item['truth_tokens'][-1,1:], item['pred_tokens'][1:]])
        patch = tokens.transpose(1,2).reshape(3,768,16,16)
        raw = patch * torch.sqrt(item['normalizer_var'] + 1e-5) + item['normalizer_mean']
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = decoder(raw.flatten(2).transpose(1,2).to(args.device), drop_cls_token=False).logits
            decoded = (decoder.unpatchify(logits).float() * image_std + image_mean).clamp(0,1)
            decoded = F.interpolate(decoded, (224,224), mode='bicubic', align_corners=False).clamp(0,1)
        decoded_np = (decoded.cpu().permute(0,2,3,1).numpy()*255).round().astype('uint8')
        true_rgb = Image.fromarray(item['rgb'][-1]).resize((224,224))
        target = np.asarray(true_rgb).astype(float)/255
        meta['decoder_truth_rgb_mae'] = float(np.abs(decoded_np[1]/255-target).mean())
        meta['prediction_rgb_mae'] = float(np.abs(decoded_np[2]/255-target).mean())
        meta['copy_source_rgb_mae'] = float(np.abs(np.asarray(Image.fromarray(item['rgb'][-2]).resize((224,224)))/255-target).mean())
        p = item['pred_tokens'][1:]
        g = item['truth_tokens'][-1,1:]
        heat = (1-F.cosine_similarity(p,g,dim=-1)).reshape(16,16).numpy()
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import colormaps
        heat_rgb = (colormaps['magma'](np.clip(heat,0,1))[...,:3]*255).astype('uint8')
        canvas = Image.new('RGB',(5*224,2*260+74),'white')
        draw = ImageDraw.Draw(canvas)
        for j in range(4):
            canvas.paste(Image.fromarray(item['rgb'][j]).resize((224,224)),(j*224,28))
            draw.text((j*224+6,8),f'Context {j+1} (old -> new)',fill='black')
        draw.text((900,45),f"Sample {meta['index']:02d} / ep {meta['episode']}\nQuery {meta['query']}\nDistance {meta['distance_m']:.2f}m\nTurn {meta['condition']['dtheta']*180/np.pi:.1f} deg\nHorizon {meta['condition']['rel_t']*128:.2f}",fill='black')
        panels = [true_rgb,Image.fromarray(decoded_np[1]),Image.fromarray(decoded_np[2]),Image.fromarray(decoded_np[0]),Image.fromarray(heat_rgb).resize((224,224),Image.Resampling.NEAREST)]
        labels = ['Target RGB (same queried pose)','Target latent decoded (control)','NWM prediction decoded','Source latent decoded (copy)','Patch error: 1-cos, dark=0 bright=1']
        for j,(panel,label) in enumerate(zip(panels,labels)):
            draw.text((j*224+4,268),label,fill='black')
            canvas.paste(panel,(j*224,288))
        draw.text((8,530),f"CLS cosine: prediction/target={meta['cls_pred_target_cos']:.3f}  source/target={meta['cls_source_target_cos']:.3f} | Patch cosine={meta['patch_pred_target_cos']:.3f}",fill='black')
        draw.text((8,554),f"RGB MAE: prediction={meta['prediction_rgb_mae']:.3f}, true-latent decode={meta['decoder_truth_rgb_mae']:.3f} | Context rerender mean abs={meta['context_rerender_mean_abs']:.6f}",fill='black')
        name = path.stem+'.jpg'
        canvas.save(out/name,quality=92)
        thumb=canvas.copy(); thumb.thumbnail((840,446)); thumbs.append(thumb)
        meta['image']=name
        records.append(meta)
        print(json.dumps(meta),flush=True)
    report={'samples':records,'count':len(records),'decoder_strict_load':True}
    fields=['cls_pred_target_cos','cls_source_target_cos','patch_pred_target_cos','patch_source_target_cos','prediction_rgb_mae','decoder_truth_rgb_mae','copy_source_rgb_mae','context_rerender_mean_abs','context_rerender_max_abs']
    report['means']={k:float(np.mean([r[k] for r in records])) for k in fields}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    # Every sample is linked, with a compact first-six overview for easy review.
    overview=Image.new('RGB',(840,446*min(6,len(thumbs))),'#eeeeee')
    for i,thumb in enumerate(thumbs[:6]): overview.paste(thumb,(0,i*446))
    overview.save(out/'overview.jpg',quality=90)
    html='''<!doctype html><meta charset="utf-8"><title>世界模型实际导航查询诊断</title>
<style>body{max-width:1200px;margin:30px auto;font:17px sans-serif;background:#f4f5f7;color:#19202c}img{width:100%}article{background:white;padding:18px;margin:22px 0}p{line-height:1.7}</style>
<h1>世界模型实际导航查询：预测与同位姿真值</h1>
<p>上排：实际输入上下文的4帧，按时间排列。下排从左到右：查询目标真实画面、真实特征解码（衡量解码器失真）、世界模型预测解码、最后上下文特征解码、预测与真值的逐图块特征误差。所有图块误差使用相同0到1色标。</p>
<p>目标是世界模型收到的候选查询位置和朝向；真实画面只用于诊断，没有送入导航决策。图片解码的是256个图块特征，导航融合使用的整体CLS特征相似度另行显示。样本按运行顺序选取，未经质量筛选；小样本不能代表完整测评集。</p>'''
    for r in records:
        html+=f'<article><h2>样本 {r["index"]} · 路线 {r["episode"]} · 候选 {r["query"]}</h2><p>距离 {r["distance_m"]:.2f} 米；整体特征相似度：预测 {r["cls_pred_target_cos"]:.3f}，直接使用源画面 {r["cls_source_target_cos"]:.3f}。</p><img loading="lazy" src="{r["image"]}"></article>'
    (out/'index.html').write_text(html)
    print(json.dumps(report['means']))


if __name__=='__main__': main()
