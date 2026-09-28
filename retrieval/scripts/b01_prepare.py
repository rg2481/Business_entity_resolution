"""User-run preparation job: fold dictionaries, labels and normalized Parquet shards.

Never starts retrieval/training. Completed shards are reused on rerun only when
the configuration, input identities and preparation code match the manifest.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time
import unicodedata

from b01_common import (ROOT,CODEX,IND,TOK,ZW,STATE_BITS,Normalizer,atomic_json,code_hash,
                       environment,fold_of,identity,latin_tokens,read_config,reference_states)
environment()
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from tqdm.auto import tqdm

SCHEMA=pa.schema([
    ('entity_id',pa.string()),('country',pa.string()),('source',pa.int8()),
    ('raw_name',pa.string()),('raw_address',pa.string()),('name_full',pa.string()),
    ('name_core',pa.string()),('address_clean',pa.string()),('owner_id',pa.string()),
    ('owner_fold',pa.int8()),('name_script',pa.int16()),('unknown_native_tokens',pa.int16()),
    ('reference_state_mask',pa.int64())])


def inputs():
    data={f'{split}_source{s}':ROOT/f'matcher/eda/pq/{split}_source{s}.parquet' for split in ('train','test') for s in (1,2,3)}
    data['truth']=ROOT/'matcher/eda/pq/train_ground_truth.parquet'
    data['language_pairs']=CODEX/'data/language/baseline_review_pairs.parquet'
    for path in data.values():
        if not path.is_file():
            raise FileNotFoundError(f'Required checked input missing: {path}')
    return data


def manifest_for(cfg,paths):
    keys=('version','validation_fold','folds','batch_rows','shard_rows')
    return {'preparation_config':{k:cfg[k] for k in keys},'inputs':{k:identity(p) for k,p in paths.items()},
            'code_sha256':code_hash('b01_common.py','b01_prepare.py')}


def make_dictionaries(path,run,cfg):
    saved=run/'dictionaries.json'
    if saved.exists():
        print('Reuse completed fold dictionaries.',flush=True)
        return json.loads(saved.read_text())
    token_all=defaultdict(Counter)
    token_fold=[defaultdict(Counter) for _ in range(cfg['folds'])]
    state_all=defaultdict(Counter)
    state_fold=[defaultdict(Counter) for _ in range(cfg['folds'])]
    state_of_bit={v:k for k,v in STATE_BITS.items()}
    file=pq.ParquetFile(path)
    with tqdm(total=file.metadata.num_rows,desc='Fit fold dictionaries',unit='rows',dynamic_ncols=True) as bar:
        for batch in file.iter_batches(batch_size=cfg['batch_rows'],columns=['s1_id','n1','n2','a1','a2']):
            d=batch.to_pydict()
            for ref,n1,n2,a1,a2 in zip(d['s1_id'],d['n1'],d['n2'],d['a1'],d['a2']):
                fold=fold_of(ref,cfg['folds'])
                if IND.search(n2):
                    a,b=latin_tokens(n1),TOK.findall(ZW.sub('',unicodedata.normalize('NFC',n2)))
                    if len(a)==len(b):
                        for latin,native in zip(a,b):
                            if IND.search(native):
                                token_all[native][latin]+=1
                                token_fold[fold][native][latin]+=1
                state=state_of_bit.get(reference_states(a1))
                if state:
                    for part in a2.split(','):
                        text=ZW.sub('',unicodedata.normalize('NFC',part.strip(' \t\r\n\"\'')))
                        if IND.search(text):
                            state_all[text][state]+=1
                            state_fold[fold][text][state]+=1
            bar.update(batch.num_rows)
    def mapping(all_counts,excluded,min_count):
        result={}
        for native,counter in all_counts.items():
            counts=counter-excluded.get(native,Counter())
            if sum(counts.values())>=min_count:
                result[native]=min(counts,key=lambda word:(-counts[word],word))
        return result
    result={'full':{'tokens':mapping(token_all,{},2),'states':mapping(state_all,{},1)},'folds':{}}
    for fold in range(cfg['folds']):
        result['folds'][str(fold)]={'tokens':mapping(token_all,token_fold[fold],2),'states':mapping(state_all,state_fold[fold],1)}
    result['method']='MD5(S1 ID)%folds; each fold dictionary excludes that fold; count ties broken lexically.'
    atomic_json(saved,result)
    return result


def load_owners(path,cfg):
    if psutil.virtual_memory().available < 4*2**30:
        raise RuntimeError('Less than 4 GiB RAM free. Stop other heavy jobs and rerun; completed shards are preserved.')
    owners={}
    file=pq.ParquetFile(path)
    with tqdm(total=file.metadata.num_rows,desc='Load training ownership',unit='S1',dynamic_ncols=True) as bar:
        for batch in file.iter_batches(batch_size=cfg['batch_rows']):
            d=batch.to_pydict()
            for ref,ids in zip(d['source1_entity_id'],d['matched_entity_ids']):
                if ids:
                    value=(ref,fold_of(ref,cfg['folds']))
                    for target in ids.split(','):
                        if target in owners:
                            raise ValueError(f'Duplicate target ownership: {target}')
                        owners[target]=value
            bar.update(batch.num_rows)
    print(f'Loaded {len(owners):,} target owners; process RAM {psutil.Process().memory_info().rss/2**30:.2f} GiB.',flush=True)
    return owners


def normalize_batch(batch,source,split,normalizer,owners,cfg):
    d=batch.to_pydict()
    out={name:[] for name in SCHEMA.names}
    for entity,name,address,country in zip(d['entity_id'],d['business_name'],d['business_address'],d['country']):
        name,address=name or '',address or ''
        full,core,unknown,mask=normalizer.name(name)
        if split=='train' and source>1:
            owner,fold=owners.get(entity,('',-1))
        else:
            owner=''
            fold=fold_of(entity,cfg['folds']) if split=='train' else -1
        vals=(entity,country,source,name,address,full,core,normalizer.address(address,country),owner,fold,mask,unknown,
              reference_states(address) if source==1 and country=='India' else 0)
        for key,value in zip(SCHEMA.names,vals):
            out[key].append(value)
    return pa.Table.from_pydict(out,schema=SCHEMA)


def prepare_one(key,path,run,normalizer,owners,cfg):
    split,src=key.split('_source')
    source=int(src)
    dest=run/'prepared'/key
    dest.mkdir(parents=True,exist_ok=True)
    file=pq.ParquetFile(path)
    total=file.metadata.num_rows
    cursor=0
    writer=None
    completed=0
    started=time.monotonic()
    with tqdm(total=total,desc=key,unit='rows',dynamic_ncols=True) as bar:
        try:
            for batch in file.iter_batches(batch_size=cfg['batch_rows']):
                pos=0
                while pos<batch.num_rows:
                    shard=cursor//cfg['shard_rows']
                    in_shard=cursor%cfg['shard_rows']
                    take=min(batch.num_rows-pos,cfg['shard_rows']-in_shard)
                    final=dest/f'part-{shard:05d}.parquet'
                    partial=dest/f'part-{shard:05d}.parquet.part'
                    expected=min(cfg['shard_rows'],total-shard*cfg['shard_rows'])
                    if in_shard==0:
                        skip=final.exists()
                        if skip and pq.ParquetFile(final).metadata.num_rows!=expected:
                            raise RuntimeError(f'Invalid completed shard: {final}')
                        if not skip:
                            writer=pq.ParquetWriter(partial,SCHEMA,compression='zstd')
                    if not skip:
                        table=normalize_batch(batch.slice(pos,take),source,split,normalizer,owners,cfg)
                        writer.write_table(table,row_group_size=cfg['batch_rows'])
                    cursor+=take
                    pos+=take
                    if cursor%cfg['shard_rows']==0 or cursor==total:
                        if not skip:
                            writer.close()
                            writer=None
                            partial.replace(final)
                        completed+=1
                        bar.set_postfix(shards=completed,cached=skip,refresh=False)
                    bar.update(take)
        finally:
            if writer is not None:
                writer.close()  # a .part file is deliberately NOT a completed checkpoint
    result={'input':identity(path),'rows':total,'shards':completed,'seconds_this_invocation':round(time.monotonic()-started,3)}
    atomic_json(dest/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--split',choices=['train','test','all'],default='all')
    args=parser.parse_args()
    cfg,run=read_config(args.config)
    pa.set_cpu_count(cfg['cpu_threads'])
    paths=inputs()
    signature=manifest_for(cfg,paths)
    run.mkdir(parents=True,exist_ok=True)
    manifest=run/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise RuntimeError('Preparation inputs/code/config changed. Choose a fresh run_dir in a copied config; old outputs were not overwritten.')
    atomic_json(manifest,signature)
    print('PREPARATION ONLY. No GPU retrieval, training, inference or submission upload will start.',flush=True)
    dictionaries=make_dictionaries(paths['language_pairs'],run,cfg)
    owners=None
    results={}
    started=time.monotonic()
    try:
        for split in ('train','test'):
            if args.split not in ('all',split):
                continue
            definition=dictionaries['full'] if split=='test' else dictionaries['folds'][str(cfg['validation_fold'])]
            normalizer=Normalizer(definition['tokens'],definition['states'])
            for source in (1,2,3):
                key=f'{split}_source{source}'
                done=run/'prepared'/key/'complete.json'
                if done.exists():
                    results[key]=json.loads(done.read_text())
                    expected=results[key]['shards']
                    actual=len(list(done.parent.glob('part-*.parquet')))
                    if actual!=expected:
                        raise RuntimeError(f'{key}: completed manifest says {expected} shards but found {actual}.')
                    print(f'Reuse completed {key}: {results[key]["rows"]:,} rows.',flush=True)
                    continue
                if split=='train' and source>1 and owners is None:
                    owners=load_owners(paths['truth'],cfg)
                results[key]=prepare_one(key,paths[key],run,normalizer,owners,cfg)
            owners=None
            gc.collect()
        atomic_json(run/'preparation_last_run.json',{'finished':datetime.now().astimezone().isoformat(),'split':args.split,
                    'elapsed_seconds':round(time.monotonic()-started,3),'files':results,'classifier_trained':False})
        print(f'PREPARATION COMPLETE: {run}\nNext: run b01_retrieval_pilot.py yourself. progress.txt is untouched.',flush=True)
    except BaseException as e:
        atomic_json(run/'preparation_last_error.json',{'error':repr(e),'time':datetime.now().astimezone().isoformat()})
        raise


if __name__=='__main__':
    main()
