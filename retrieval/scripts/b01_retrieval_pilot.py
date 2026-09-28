"""User-run GPU retrieval pilot: sampled targets versus ALL S1 in one country.

This measures retrieval recall/throughput, not a trained matcher's macro-F0.5.
No model training, final predictions, submission upload or progress.txt edits.
"""
import argparse
from collections import Counter
import gc
import hashlib
import heapq
import json
import time

from b01_common import (ROOT,CODEX,SCRIPT_NAMES,allowed_states,atomic_json,code_hash,environment,identity,read_config)
environment()
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import torch


def save_sparse(path,matrix):
    partial=path.with_suffix(path.suffix+'.part')
    with partial.open('wb') as f:
        sp.save_npz(f,matrix,compressed=False)
    partial.replace(path)


def save_table(path,table):
    partial=path.with_suffix(path.suffix+'.part')
    pq.write_table(table,partial,compression='zstd')
    partial.replace(path)


def prepared(run,split,source):
    folder=run/'prepared'/f'{split}_source{source}'
    done=folder/'complete.json'
    if not done.is_file():
        raise RuntimeError(f'Preparation is not complete for {folder.name}. Run b01_prepare.py first.')
    files=sorted(folder.glob('part-*.parquet'))
    if len(files)!=json.loads(done.read_text())['shards']:
        raise RuntimeError(f'Missing shards under {folder}')
    return files


def dataset(paths):
    return ds.dataset([str(p) for p in paths],format='parquet')


def vectorizer(cfg):
    return HashingVectorizer(analyzer='char_wb',ngram_range=(3,3),n_features=cfg['hash_features'],
                             alternate_sign=False,norm=None,lowercase=False,dtype=np.float32)


def binary_vectors(hv,texts):
    x=hv.transform(texts).tocsr()
    x.sum_duplicates()
    x.data[:]=1.0
    x.indices=x.indices.astype(np.int32,copy=False)
    x.indptr=x.indptr.astype(np.int32,copy=False)
    return x


def weighted_vectors(x,idf):
    x.data*=idf[x.indices]
    return normalize(x,copy=False)


def index_for(run,cfg,split,country):
    paths=prepared(run,split,1)
    directory=run/'indexes'/f'{split}_{country}'
    directory.mkdir(parents=True,exist_ok=True)
    signature={'prepared':[identity(p) for p in paths],'features':cfg['hash_features'],'batch_rows':cfg['batch_rows'],
               'code':code_hash('b01_common.py','b01_retrieval_pilot.py')}
    manifest=directory/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise RuntimeError(f'Index configuration changed: use a fresh run_dir. Existing index retained: {directory}')
    atomic_json(manifest,signature)
    ready=directory/'complete.json'
    if not ready.exists():
        data=dataset(paths)
        filt=ds.field('country')==country
        print(f'Count reference rows for {split}/{country} (Parquet metadata/filter pass).',flush=True)
        total=data.count_rows(filter=filt)
        if not total:
            raise ValueError(f'No references for {split}/{country}')
        hv=vectorizer(cfg)
        name_parts,address_parts,meta_parts=[],[],[]
        df_name=np.zeros(cfg['hash_features'],dtype=np.int64)
        df_address=np.zeros_like(df_name)
        scanner=data.scanner(columns=['entity_id','name_core','address_clean','reference_state_mask'],filter=filt,
                             batch_size=cfg['batch_rows'],use_threads=False)
        with tqdm(total=total,desc=f'Index {country}',unit='refs',dynamic_ncols=True) as bar:
            for number,batch in enumerate(scanner.to_batches()):
                if not batch.num_rows:
                    continue
                d=batch.to_pydict()
                prefix=directory/f'chunk-{number:05d}'
                pn=prefix.with_suffix('.name.npz')
                pa_=prefix.with_suffix('.address.npz')
                pm=prefix.with_suffix('.metadata.parquet')
                pc=prefix.with_suffix('.complete.json')
                digest=hashlib.sha256('\n'.join(d['entity_id']).encode()).hexdigest()
                stamp={'rows':batch.num_rows,'id_hash':digest}
                if pc.exists():
                    if json.loads(pc.read_text())!=stamp:
                        raise RuntimeError('Reference order changed; index checkpoint cannot be reused.')
                    xn,xa=sp.load_npz(pn),sp.load_npz(pa_)
                    meta=pq.read_table(pm)
                else:
                    xn,xa=binary_vectors(hv,d['name_core']),binary_vectors(hv,d['address_clean'])
                    meta=pa.table({'entity_id':d['entity_id'],'reference_state_mask':pa.array(d['reference_state_mask'],type=pa.int64())})
                    save_sparse(pn,xn)
                    save_sparse(pa_,xa)
                    save_table(pm,meta)
                    atomic_json(pc,stamp)
                name_parts.append(xn)
                address_parts.append(xa)
                meta_parts.append(meta)
                df_name+=np.bincount(xn.indices,minlength=cfg['hash_features'])
                df_address+=np.bincount(xa.indices,minlength=cfg['hash_features'])
                bar.update(batch.num_rows)
        # These compact phases are opaque library calls; report them honestly.
        print('Assemble CSR matrices and reference-only IDF weights; next progress bar starts after these library calls.',flush=True)
        xn=sp.vstack(name_parts,format='csr')
        del name_parts
        xa=sp.vstack(address_parts,format='csr')
        del address_parts
        name_idf=(np.log((total+1)/(df_name+1))+1).astype(np.float32)
        address_idf=(np.log((total+1)/(df_address+1))+1).astype(np.float32)
        xn=weighted_vectors(xn,name_idf)
        xa=weighted_vectors(xa,address_idf)
        with tqdm(total=4,desc='Save reference index',unit='files',dynamic_ncols=True) as bar:
            save_sparse(directory/'name.npz',xn); bar.update(1)
            save_sparse(directory/'address.npz',xa); bar.update(1)
            save_table(directory/'metadata.parquet',pa.concat_tables(meta_parts)); bar.update(1)
            partial=directory/'idf.npz.part'
            with partial.open('wb') as f:
                np.savez(f,name=name_idf,address=address_idf)
            partial.replace(directory/'idf.npz'); bar.update(1)
        atomic_json(ready,{'rows':total,'name_nnz':xn.nnz,'address_nnz':xa.nnz,'idf_fit':'country-specific reference text only; no labels'})
        del xn,xa,meta_parts
        gc.collect()
    return directory


def sample_queries(run,cfg,split,country,directory):
    saved=directory/'queries.parquet'
    if saved.exists():
        return pq.read_table(saved).to_pylist()
    paths=prepared(run,split,2)+prepared(run,split,3)
    data=dataset(paths)
    filt=ds.field('country')==country
    print(f'Count target rows for {split}/{country} before deterministic sampling.',flush=True)
    total=data.count_rows(filter=filt)
    columns=['entity_id','country','source','raw_name','raw_address','name_core','address_clean','owner_id','owner_fold','name_script','unknown_native_tokens']
    scanner=data.scanner(columns=columns,filter=filt,batch_size=cfg['batch_rows'],use_threads=False)
    heaps={}
    rng=np.random.default_rng(cfg['seed'])
    seen=Counter()
    counter=0
    with tqdm(total=total,desc=f'Sample {country}',unit='targets',dynamic_ncols=True) as bar:
        for batch in scanner.to_batches():
            d=batch.to_pydict()
            priorities=rng.random(batch.num_rows)
            for i in range(batch.num_rows):
                if split=='test':
                    kind='unlabeled'
                elif not d['owner_id'][i]:
                    kind='distractor'
                elif d['owner_fold'][i]==cfg['validation_fold']:
                    kind='heldout_match'
                else:
                    continue
                seen[kind]+=1
                heap=heaps.setdefault(kind,[])
                key=float(priorities[i])
                if len(heap)<cfg['queries_per_class'] or key < -heap[0][0]:
                    row={column:d[column][i] for column in columns}
                    row['query_kind']=kind
                    item=(-key,counter,row)
                    counter+=1
                    if len(heap)<cfg['queries_per_class']:
                        heapq.heappush(heap,item)
                    else:
                        heapq.heapreplace(heap,item)
            bar.update(batch.num_rows)
    rows=sorted((item[2] for heap in heaps.values() for item in heap),key=lambda r:r['entity_id'])
    if not rows:
        raise ValueError('No eligible queries.')
    save_table(saved,pa.Table.from_pylist(rows))
    atomic_json(directory/'sampling.json',{'total_targets':total,'eligible_by_kind':dict(seen),'sampled_by_kind':dict(Counter(r['query_kind'] for r in rows)),
                                         'method':'Uniform random priorities per target, smallest N per class; fixed seed; no labels used in vectorizer.'})
    return rows


def gpu_csr(x):
    return torch.sparse_csr_tensor(torch.from_numpy(x.indptr.astype(np.int32)),torch.from_numpy(x.indices.astype(np.int32)),
                                  torch.from_numpy(x.data),size=x.shape,device='cuda')


def dense_queries(x):
    # Densify only a query minibatch, never the full source table.
    return torch.from_numpy(x.toarray().T.copy()).to('cuda')


def retrieve_batch(pn,pa_,qn,qa,ref_states,ref_ambiguous,query_masks,cfg):
    with torch.inference_mode():
        sn=torch.sparse.mm(pn,dense_queries(qn))
        sa=torch.sparse.mm(pa_,dense_queries(qa))
        if cfg['state_filter']:
            allowed=torch.tensor(query_masks,dtype=torch.int64,device='cuda')
            keep=((ref_states[:,None]&allowed[None,:])!=0)|(allowed[None,:]==0)|ref_ambiguous[:,None]
        else:
            keep=None
        result={}
        k=min(cfg['top_k'],sn.shape[0])
        for mode,scores in [('name',sn),('address',sa),('combo',sn*cfg['name_weight']+sa*(1-cfg['name_weight']))]:
            # Original evidence matrices remain unchanged so retrieved scores can
            # be compared and inspected even when the state filter is active.
            ranked=scores.masked_fill(~keep,float('-inf')) if keep is not None else scores
            values,indices=torch.topk(ranked,k,dim=0)
            result[mode]=(indices.T.cpu().numpy(),values.T.cpu().numpy(),
                          sn.gather(0,indices).T.cpu().numpy(),sa.gather(0,indices).T.cpu().numpy())
        return result


def summarize(directory,rows,elapsed,processed):
    details=[]
    for path in sorted(directory.glob('batch-*.summary.parquet')):
        details.extend(pq.read_table(path).to_pylist())
    if len(details)!=len(rows):
        raise RuntimeError('Incomplete retrieval summaries; rerun to resume remaining batches.')
    matched=[r for r in details if r['query_kind']=='heldout_match']
    metrics={}
    for mode in ('name','address','combo'):
        metrics[mode]={str(k):sum(0<r[f'{mode}_true_rank']<=k for r in matched)/len(matched) if matched else None for k in (1,5,10)}
    slices={}
    for label in ['Latin']+SCRIPT_NAMES:
        selected=[r for r in matched if r['script']==label]
        if selected:
            slices[label]={'queries':len(selected),'combo_recall_at_10':sum(0<r['combo_true_rank']<=10 for r in selected)/len(selected)}
    sampling=json.loads((directory/'sampling.json').read_text())
    per_query=elapsed/processed if processed else None
    report={'query_count':len(rows),'heldout_matched_queries':len(matched),'retrieval_recall_at_k':metrics,'script_slices':slices,
            'seconds_per_query_this_invocation':per_query,'gpu_search_and_checkpoint_seconds_this_invocation':elapsed,
            'rough_full_country_search_hours':per_query*sampling['total_targets']/3600 if per_query else None,
            'scope':'Sampled target-level retrieval diagnostic, with full country reference pool. Not macro-F0.5 or a trained-model result.',
            'timing_caveat':'Linear projection from measured batches includes checkpoint writes, excludes preparation/indexing, and is not a guaranteed completion time.'}
    atomic_json(directory/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--split',choices=['train','test'],default='train')
    parser.add_argument('--country',choices=['India','US','France'],default='India')
    args=parser.parse_args()
    cfg,run=read_config(args.config)
    if args.country=='France' and args.split=='train':
        parser.error('France exists only in test.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this terminal. Run the supplied preflight in your normal WSL terminal; no CPU brute-force fallback is attempted.')
    torch.set_num_threads(cfg['cpu_threads'])
    pa.set_cpu_count(cfg['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0)
    if not 0<cfg['name_weight']<1 or cfg['top_k']<10:
        raise ValueError('Use 0<name_weight<1 and top_k>=10 for the pilot metrics.')
    expected=json.loads((run/'manifest.json').read_text())
    if expected['code_sha256']!=code_hash('b01_common.py','b01_prepare.py'):
        raise RuntimeError('Preparation code changed since this run. Use a new run_dir rather than mixing normalized data versions.')
    if cfg['validation_fold']!=expected['preparation_config']['validation_fold']:
        raise RuntimeError('Validation fold differs from preparation; use a fresh run_dir for another fold.')
    index=index_for(run,cfg,args.split,args.country)
    request={k:cfg[k] for k in ('hash_features','query_batch','queries_per_class','top_k','seed','state_filter','name_weight','validation_fold')}
    request.update(split=args.split,country=args.country,code=code_hash('b01_common.py','b01_retrieval_pilot.py'))
    tag=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:12]
    directory=run/'pilot'/f'{args.split}_{args.country}_{tag}'
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'config.json',request)
    rows=sample_queries(run,cfg,args.split,args.country,directory)
    if (directory/'report.json').exists() and all(
        (directory/f'batch-{start:07d}.summary.parquet').exists() and (directory/f'batch-{start:07d}.candidates.parquet').exists()
        for start in range(0,len(rows),cfg['query_batch'])):
        print('Reuse completed pilot; its measured timings are preserved.',flush=True)
        print((directory/'report.json').read_text(),flush=True)
        return
    print('Load cached CSR index and transfer it to CUDA; no full-source dense matrix is created.',flush=True)
    with tqdm(total=4,desc='Load GPU index',unit='parts',dynamic_ncols=True) as bar:
        pn=gpu_csr(sp.load_npz(index/'name.npz')); bar.update(1)
        pa_=gpu_csr(sp.load_npz(index/'address.npz')); bar.update(1)
        meta=pq.read_table(index/'metadata.parquet').to_pydict(); bar.update(1)
        with np.load(index/'idf.npz') as weights:
            name_idf,address_idf=weights['name'],weights['address']
        bar.update(1)
    ref_ids=meta['entity_id']
    ref_states=torch.tensor(meta['reference_state_mask'],dtype=torch.int64,device='cuda')
    # Preserve unknown and ambiguous reference states instead of guessing.
    ref_ambiguous=torch.tensor([m==0 or int(m).bit_count()>1 for m in meta['reference_state_mask']],dtype=torch.bool,device='cuda')
    hv=vectorizer(cfg)
    elapsed=0.0
    processed=0
    with tqdm(total=len(rows),desc=f'GPU retrieval {args.country}',unit='queries',dynamic_ncols=True) as bar:
        for start in range(0,len(rows),cfg['query_batch']):
            queries=rows[start:start+cfg['query_batch']]
            output=directory/f'batch-{start:07d}.candidates.parquet'
            summary=directory/f'batch-{start:07d}.summary.parquet'
            if summary.exists() and output.exists():
                bar.update(len(queries))
                continue
            torch.cuda.synchronize()
            began=time.monotonic()
            qn=weighted_vectors(binary_vectors(hv,[q['name_core'] for q in queries]),name_idf)
            qa=weighted_vectors(binary_vectors(hv,[q['address_clean'] for q in queries]),address_idf)
            result=retrieve_batch(pn,pa_,qn,qa,ref_states,ref_ambiguous,[allowed_states(q['name_script']) for q in queries],cfg)
            candidates,summaries=[],[]
            for i,q in enumerate(queries):
                script='Latin' if not q['name_script'] else SCRIPT_NAMES[int(q['name_script']).bit_length()-1]
                detail={'query_id':q['entity_id'],'query_kind':q['query_kind'],'source':q['source'],'script':script,'true_s1_id':q['owner_id']}
                for mode,(indices,values,names,addresses) in result.items():
                    true_rank=-1
                    for rank,(ref,score,ns,ad) in enumerate(zip(indices[i],values[i],names[i],addresses[i]),1):
                        if not np.isfinite(score):
                            continue
                        owner=ref_ids[int(ref)]
                        if owner==q['owner_id']:
                            true_rank=rank
                        candidates.append({'query_id':q['entity_id'],'candidate_s1_id':owner,'mode':mode,'rank':rank,
                                           'name_score':float(ns),'address_score':float(ad),'retrieval_score':float(score),
                                           'query_kind':q['query_kind'],'true_s1_id':q['owner_id']})
                    detail[f'{mode}_true_rank']=true_rank
                summaries.append(detail)
            # Explicit schema permits a legitimate zero-candidate batch.
            cand_schema=pa.schema([('query_id',pa.string()),('candidate_s1_id',pa.string()),('mode',pa.string()),('rank',pa.int32()),
                                  ('name_score',pa.float32()),('address_score',pa.float32()),('retrieval_score',pa.float32()),
                                  ('query_kind',pa.string()),('true_s1_id',pa.string())])
            save_table(output,pa.Table.from_pylist(candidates,schema=cand_schema))
            save_table(summary,pa.Table.from_pylist(summaries))
            torch.cuda.synchronize()
            elapsed+=time.monotonic()-began
            processed+=len(queries)
            bar.set_postfix(vram_GiB=f'{torch.cuda.max_memory_allocated()/2**30:.2f}',refresh=False)
            bar.update(len(queries))
    summarize(directory,rows,elapsed,processed)
    print(f'PILOT COMPLETE: {directory}/report.json\nNo classifier training or submission was started. progress.txt is untouched.',flush=True)


if __name__=='__main__':
    main()
