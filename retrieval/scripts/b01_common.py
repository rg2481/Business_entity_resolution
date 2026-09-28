"""Shared, import-safe B01 configuration and identity-preserving normalization."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import re
import sys
import unicodedata
sys.dont_write_bytecode = True
import regex

ROOT = Path(__file__).resolve().parents[2]
CODEX = ROOT / 'retrieval'
TOK = re.compile(r'[\w\u0900-\u0d7f]+')
IND = re.compile(r'[\u0900-\u0d7f]')
ZW = re.compile('[\u200c\u200d]')
LATIN_RUN = regex.compile(r'\p{Latin}[\p{Latin}\p{M}]*')
NOT_WORD = regex.compile(r'[^\p{L}\p{M}\p{N}]+')
LEGAL = set('inc incorporated llc corp corporation co company ltd limited pvt private llp plc lp pllc pc sarl sas sasu eurl sa sci snc ets ei'.split())
DBA = re.compile(r'(?i)^.*?\b(?:d/b/a|dba|doing business as|formerly known as|formerly|f/k/a|fka|trading as|t/a|also known as|a\.k\.a\.?|aka)\b:?\s*')
STATES = ['Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh','Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand',
          'Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim',
          'Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal','Andaman and Nicobar Islands','Chandigarh',
          'Dadra and Nagar Haveli','Daman and Diu','Delhi','Jammu and Kashmir','Ladakh','Lakshadweep','Puducherry']
STATE_BITS = {s: 1 << i for i,s in enumerate(STATES)}
STATE_ALIASES = {s.lower():s for s in STATES}
STATE_ALIASES.update({'orissa':'Odisha','pondicherry':'Puducherry','keralam':'Kerala'})
STATE_CODES = {'ap':'Andhra Pradesh','br':'Bihar','dl':'Delhi','gj':'Gujarat','hr':'Haryana','ka':'Karnataka','kl':'Kerala','mp':'Madhya Pradesh',
               'mh':'Maharashtra','od':'Odisha','or':'Odisha','pb':'Punjab','rj':'Rajasthan','tn':'Tamil Nadu','tg':'Telangana','ts':'Telangana',
               'up':'Uttar Pradesh','wb':'West Bengal'}
SCRIPT_STATES = [
    ['Maharashtra','Delhi','Uttar Pradesh','Haryana','Rajasthan','Bihar','Madhya Pradesh'], ['West Bengal'], ['Punjab'], ['Gujarat'],
    ['Odisha'], ['Tamil Nadu'], ['Andhra Pradesh','Telangana'], ['Karnataka'], ['Kerala']]
SCRIPT_NAMES = ['Devanagari','Bengali','Gurmukhi','Gujarati','Odia','Tamil','Telugu','Kannada','Malayalam']
SCRIPT_STATE_MASKS = [sum(STATE_BITS[s] for s in ss) for ss in SCRIPT_STATES]
FR_REGIONS = {'nord':'Hauts-de-France','gironde':'Nouvelle-Aquitaine','loire-atlantique':'Pays de la Loire'}
US_STATES = dict(zip(
    ['alabama','alaska','arizona','arkansas','california','colorado','connecticut','delaware','florida','georgia','hawaii','idaho',
     'illinois','indiana','iowa','kansas','kentucky','louisiana','maine','maryland','massachusetts','michigan','minnesota',
     'mississippi','missouri','montana','nebraska','nevada','new hampshire','new jersey','new mexico','new york','north carolina',
     'north dakota','ohio','oklahoma','oregon','pennsylvania','rhode island','south carolina','south dakota','tennessee','texas',
     'utah','vermont','virginia','washington','west virginia','wisconsin','wyoming','district of columbia'],
    'al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc'.split()))
STREET = {'street':'st','road':'rd','avenue':'ave','boulevard':'blvd','drive':'dr','lane':'ln','court':'ct','circle':'cir',
          'apartment':'apt','suite':'ste','floor':'fl','number':'no'}


def environment():
    values = {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'cache','PIP_CACHE_DIR':'cache/pip','HF_HOME':'cache/huggingface',
              'TORCH_HOME':'cache/torch','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda','MPLCONFIGDIR':'cache/matplotlib',
              'NUMBA_CACHE_DIR':'cache/numba','JOBLIB_TEMP_FOLDER':'tmp','POLARS_TEMP_DIR':'tmp'}
    for key, rel in values.items():
        path=CODEX/rel
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PIP_DISABLE_PIP_VERSION_CHECK']='1'


def read_config(path=None):
    path=Path(path) if path else CODEX/'configs/b01.json'
    cfg=json.loads(path.read_text())
    run=(ROOT/cfg['run_dir']).resolve()
    if not run.is_relative_to(CODEX.resolve()):
        raise ValueError('run_dir must stay inside retrieval/')
    if cfg['shard_rows']%cfg['batch_rows']:
        raise ValueError('shard_rows must be a multiple of batch_rows')
    return cfg,run


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part')
    partial.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    partial.replace(path)


def identity(path):
    path=Path(path)
    s=path.stat()
    return {'path':str(path.relative_to(ROOT)),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def code_hash(*names):
    return hashlib.sha256(b''.join((CODEX/'scripts'/n).read_bytes() for n in names)).hexdigest()


def fold_of(entity_id,folds=5):
    return int(hashlib.md5(entity_id.encode()).hexdigest(),16)%folds


def latin_tokens(text):
    return re.findall(r'[a-z0-9]+',unicodedata.normalize('NFKD',text.lower()).encode('ascii','ignore').decode())


@lru_cache(maxsize=30000)
def fold_latin(text):
    return LATIN_RUN.sub(lambda m: ''.join(ch for ch in unicodedata.normalize('NFKD',m.group()) if not unicodedata.category(ch).startswith('M')),text)


def clean_words(text):
    return ' '.join(NOT_WORD.sub(' ',fold_latin(unicodedata.normalize('NFC',text)).lower()).split())


def reference_states(address):
    mask=0
    for part in address.split(','):
        key=part.strip(' \t\r\n\"\'').lower().replace('&','and')
        s=STATE_ALIASES.get(key)
        if s:
            mask |= STATE_BITS[s]
    return mask


def script_mask(text):
    mask=0
    for ch in IND.findall(text):
        mask |= 1 << ((ord(ch)-0x0900)//128)
    return mask


def allowed_states(mask):
    return _allowed(mask) if mask else 0


def _allowed(mask):
    result=0
    for i, states in enumerate(SCRIPT_STATE_MASKS):
        if mask & (1<<i):
            result |= states
    return result


class Normalizer:
    def __init__(self,dictionary,state_aliases):
        self.dictionary=dictionary
        self.state_aliases=state_aliases

    def name(self,raw):
        raw=raw or ''
        text=ZW.sub('',unicodedata.normalize('NFC',raw))
        native=[t for t in TOK.findall(text) if IND.search(t)]
        unknown=sum(t not in self.dictionary for t in native)
        text=TOK.sub(lambda m:self.dictionary.get(m.group(),m.group()),text)
        text=DBA.sub('',text,count=1).split(' | ',1)[0]
        text=re.sub(r'(?i)(^[@#]+|www\.|\.(?:co\.in|com|in|net|org|fr)\b)',' ',text)
        full=clean_words(text)
        full=re.sub(r'\bl l c\b','llc',full)
        full=re.sub(r'\bl l p\b','llp',full)
        core=' '.join(t for t in full.split() if t not in LEGAL)
        return full,core or full,unknown,script_mask(raw)

    def address(self,raw,country):
        parts=[]
        for comp in (raw or '').split(','):
            text=ZW.sub('',unicodedata.normalize('NFC',comp.strip(' \t\r\n\"\'')))
            if country=='India':
                text=self.state_aliases.get(text,STATE_ALIASES.get(text.lower(),STATE_CODES.get(text.lower(),text)))
            elif country=='France':
                text=FR_REGIONS.get(text.lower(),text)
            elif country=='US':
                text=US_STATES.get(text.lower(),text)
            parts.append(text)
        text=clean_words(', '.join(parts))
        mapping=STREET if country!='France' else {'bd':'boulevard','boul':'boulevard','av':'avenue','all':'allee','r':'rue'}
        return ' '.join(mapping.get(t,t) for t in text.split() if t!='null')
