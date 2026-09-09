#!/usr/bin/env python3
"""Decode held-out predictions and show deterministic representative queries."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True)
    ap.add_argument('--upstream',required=True);ap.add_argument('--name',default='confirmation')
    args=ap.parse_args();root=Path(args.root);up=Path(args.upstream)
    review=json.loads((root/'review.json').read_text());winner=review['winner'];paired=review['paired_queries']
    chosen=[]
    for scene in sorted({x['scene'] for x in paired}):
        items=sorted([x for x in paired if x['scene']==scene],key=lambda x:x['delta_cls_cosine'])
        chosen.append((items[len(items)//2],'scene median improvement'))
    for item,label in [(max(paired,key=lambda x:x['delta_cls_cosine']),'largest improvement'),
                       (min(paired,key=lambda x:x['delta_cls_cosine']),'largest regression')]:
        if item['id'] not in {x['id'] for x,_ in chosen}:chosen.append((item,label))
    ids={x['id'] for x,_ in chosen};predictions={}
    for file in sorted((root/(args.name+'_predictions')).glob('*.pt')):
        data=torch.load(file,map_location='cpu',weights_only=False)
        for i,row in enumerate(data['rows']):
            if row['id'] in ids and row['seed']==11:predictions[(row['id'],row['mode'])]=data['tokens'][i].float()
    rows=json.loads((root/(args.name+'.json')).read_text())['rows'];lookup={(r['id'],r['mode'],r['seed']):r for r in rows}
    from vlnce_baselines.nwm.panorama_context import observed_view
    from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
    norm=RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt')
    sys.path.insert(0,str(up/'RAE/src/stage1'));decoder_module=importlib.import_module('decoders')
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(str(up/'RAE/configs/decoder/ViTXL'))
    config.hidden_size,config.patch_size,config.image_size=768,16,256
    decoder=decoder_module.GeneralDecoder(config,num_patches=256)
    decoder.load_state_dict(torch.load(up/'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',map_location='cpu'),strict=True)
    decoder.requires_grad_(False).eval().cuda()
    processor=json.loads((up/'models/encoders/dinov2-with-registers-base/preprocessor_config.json').read_text())
    mean=torch.tensor(processor['image_mean'],device='cuda').view(1,3,1,1)
    std=torch.tensor(processor['image_std'],device='cuda').view(1,3,1,1)
    out=root/'visuals';out.mkdir(exist_ok=True);records=[]
    for index,(item,label) in enumerate(chosen):
        key=item['id'];row=lookup[(key,winner,11)]
        encoded=torch.load(row['encoded_file'],map_location='cpu',weights_only=False)
        qi=next(i for i,q in enumerate(encoded['queries']) if q['id']==key);q=encoded['queries'][qi]
        capture=torch.load(encoded['source'],map_location='cpu',weights_only=False)
        tokens=torch.stack([predictions[(key,'front')],predictions[(key,winner)],encoded['truth'][qi].float()])[:,1:]
        patch=tokens.transpose(1,2).reshape(3,768,16,16)
        raw=patch*torch.sqrt(norm.var+norm.eps)+norm.mean
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=decoder(raw.flatten(2).transpose(1,2).cuda(),drop_cls_token=False).logits
            decoded=(decoder.unpatchify(logits).float()*std+mean).clamp(0,1)
            decoded=F.interpolate(decoded,(224,224),mode='bicubic',align_corners=False).clamp(0,1)
        rgb=(decoded.cpu().permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
        plan=q['variants'][winner];source=plan['source_index']
        source_rgb=observed_view(capture['cube_rgb'][source],plan['view_yaws'][source],capture['native_world_rgb12'][source])
        panels=[capture['front_rgb'][-1],source_rgb,capture['target_rgb'][qi],*rgb]
        titles=['Original front source','Selected panorama source','Target RGB (fixed position/yaw)',
                'Original NWM prediction','Selected NWM prediction','Target latent decoded (control)']
        image=Image.new('RGB',(672,604),'white');draw=ImageDraw.Draw(image)
        draw.text((8,8),f"{key} | {label}\nscene {item['scene']} | original turn {item['original_turn_deg']:.1f} deg",fill='black')
        for j,(panel,title) in enumerate(zip(panels,titles)):
            x=(j%3)*224;y=60+(j//3)*254
            draw.text((x+4,y),title,fill='black');image.paste(Image.fromarray(panel),(x,y+20))
        base=lookup[(key,'front',11)]
        draw.text((8,576),f"Seed 11 CLS cosine: {base['cls_cosine']:.3f} -> {row['cls_cosine']:.3f} | source frame {source}",fill='black')
        filename=f'example_{index:02d}.jpg';image.save(out/filename,quality=94)
        records.append({**item,'selection_reason':label,'image':filename,'display_seed':11,
            'display_base_cls':base['cls_cosine'],'display_winner_cls':row['cls_cosine']})
    html='''<!doctype html><meta charset="utf-8"><title>全景上下文：独立场景预测质量</title>
<style>body{max-width:1000px;margin:30px auto;background:#f4f5f7;font:18px sans-serif;color:#202733}p{line-height:1.7}article{background:white;padding:20px;margin:24px 0}img{max-width:100%}</style>
<h1>全景上下文预测质量复核</h1><p>每个复核场景选取改善幅度居中的一个查询，另展示最大改善及最大退化。选图规则不参与方案选择。上排：原前置画面、选定全景视图、固定目标真值；下排：原预测、新预测、真值特征解码。图片使用种子11，报告指标另外平均3个种子。</p>'''
    for r in records:html+=f'<article><h2>{r["id"]}</h2><p>{r["selection_reason"]}；原转角 {r["original_turn_deg"]:.1f}°；3种子平均CLS：{r["base_cls_cosine"]:.3f} → {r["winner_cls_cosine"]:.3f}。</p><img src="{r["image"]}"></article>'
    (out/'index.html').write_text(html);(out/'examples.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'examples':len(records),'output':str(out)}),flush=True)


if __name__=='__main__':main()
