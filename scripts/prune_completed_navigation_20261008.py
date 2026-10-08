"""One-off audited cleanup of three completed navigation experiments only."""
import argparse,json,re,hashlib,os
from pathlib import Path
BASE=Path('/mnt/data2tb/ETP-R1_data/experiments')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','apply']);ap.add_argument('--audit',required=True);a=ap.parse_args();audit=Path(a.audit)
 if a.action=='plan':
  summaries={k:json.loads((audit/(k+'.json')).read_text()) for k in ['persistent_server','persistent_eval','direct','rxr']}
  assert all(d['status']=='completed' for d in summaries.values())
  groups=[('ghost_concat_persistent_compiled_10k_20260911',summaries['persistent_server']['results']+summaries['persistent_eval']['results']),('ghost_concat_direct_compiled_10k_20260909',summaries['direct']['results'])]
  rr=[dict(iteration=j['iteration'],episodes=j['validation']['episodes'],metrics=j['validation']['metrics']) for j in summaries['rxr']['jobs'].values() if j['status']=='completed' and j['exit_code']==0]
  assert len(rr)==149
  groups.append(('rxr_dino_baseline_20260920',rr));plan=[]
  for name,results in groups:
   assert all(r['episodes']==(11006 if name.startswith('rxr') else 1839) for r in results)
   keep=set();reasons={}
   for metric in ['success','spl','ndtw','sdtw']:
    r=max(results,key=lambda r:(r['metrics'][metric],-r['iteration']));keep.add(r['iteration']);reasons.setdefault(str(r['iteration']),[]).append('best_'+metric)
   for r in sorted(results,key=lambda r:(r['metrics']['success']+r['metrics']['spl'],-r['iteration']),reverse=True):
    if len(keep)>=5:break
    keep.add(r['iteration']);reasons.setdefault(str(r['iteration']),[]).append('success_plus_spl')
   root=BASE/name;files=[]
   for dirname in ['train','train_fast_resume_20260921']:
    files.extend((root/dirname).glob('**/checkpoints/**/ckpt.iter*.pth'))
   completed={r['iteration'] for r in results}
   for p in sorted(files):
    iteration=int(re.search(r'iter(\d+)',p.name)[1]);st=p.stat()
    delete=iteration in completed and iteration not in keep
    plan.append(dict(path=str(p),iteration=iteration,action='delete' if delete else 'keep',size=st.st_size,inode=st.st_ino,device=st.st_dev,mtime_ns=st.st_mtime_ns,links=st.st_nlink,experiment=name))
   (audit/(name+'_selection.json')).write_text(json.dumps(dict(keep=sorted(keep),reasons=reasons,metrics={str(r['iteration']):r['metrics'] for r in results if r['iteration'] in keep}),indent=2))
  (audit/'plan.json').write_text(json.dumps(plan,indent=2))
  print(json.dumps(dict(delete_files=sum(r['action']=='delete' for r in plan),logical_gib=sum(r['size'] for r in plan if r['action']=='delete')/1024**3,keep_files=sum(r['action']=='keep' for r in plan))))
 else:
  plan=json.loads((audit/'plan.json').read_text());targets={r['path'] for r in plan if r['action']=='delete'}
  assert not (audit/'deleted.jsonl').exists()
  # Reject source symlinks and any changed file; protect current worktree links.
  protected=set()
  roots=[Path('/home/gwl/project/etpr1/ETP-R1'),Path('/home/gwl/project/etpr1/ETP-R1-stage2-e24'),Path('/home/gwl/project/etpr1/ETP-R1-recursive-future-rollout')]
  for root in roots:
   for d,dirs,files in os.walk(root,followlinks=False):
    dirs[:]=[x for x in dirs if x not in ['.git','.runtime','vendor','__pycache__'] and not Path(d,x).is_symlink()]
    for f in files:
     p=Path(d,f)
     if p.is_symlink():protected.add(str(p.resolve()))
  assert not targets&protected,targets&protected
  for r in plan:
   p=Path(r['path']);assert p.is_relative_to(BASE) and not p.is_symlink();st=p.stat()
   assert (st.st_size,st.st_ino,st.st_dev,st.st_mtime_ns)==(r['size'],r['inode'],r['device'],r['mtime_ns'])
  keep_hashes={r['path']:digest(Path(r['path'])) for r in plan if r['action']=='keep'}
  (audit/'kept_sha256.json').write_text(json.dumps(keep_hashes,indent=2))
  with (audit/'deleted.jsonl').open('x') as log:
   for r in plan:
    if r['action']=='delete':Path(r['path']).unlink();log.write(json.dumps(r)+'\n');log.flush()
  assert all(Path(p).exists() for p in keep_hashes)
  assert all(not Path(p).exists() for p in targets)
  print('DELETED',len(targets),'KEPT',len(keep_hashes))
if __name__=='__main__':main()
