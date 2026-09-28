"""Short hardware/environment check; no dataset scan or performance benchmark."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from b01_common import ROOT,CODEX,atomic_json,environment
environment()
import psutil
from tqdm.auto import tqdm


def main():
    result={}
    with tqdm(total=4,desc='Preflight',unit='checks',dynamic_ncols=True) as bar:
        result.update(kernel=platform.release(),logical_cpus=os.cpu_count(),
                      wsl_ram_gib=round(psutil.virtual_memory().total/2**30,2),
                      available_ram_gib=round(psutil.virtual_memory().available/2**30,2),
                      free_repo_gib=round(shutil.disk_usage(ROOT).free/2**30,2))
        result['cpu']=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
        bar.update(1)
        result['packages']={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pyarrow','torch','tqdm','regex','psutil')}
        try:
            probe=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv'],capture_output=True,text=True,timeout=8)
            result['nvidia_smi']=probe.stdout.strip() if probe.returncode==0 else probe.stderr.strip()
        except (OSError,subprocess.TimeoutExpired) as e:
            result['nvidia_smi']=repr(e)
        bar.update(1)
        import torch
        result['cuda_available']=torch.cuda.is_available()
        result['torch_cuda']=torch.version.cuda
        if result['cuda_available']:
            result['gpu']=torch.cuda.get_device_name(0)
            result['compiled_arches']=torch.cuda.get_arch_list()
            result['capability']=list(torch.cuda.get_device_capability(0))
            # Two nonzeros: verifies the sparse CUDA operation used by retrieval.
            x=torch.sparse_csr_tensor(torch.tensor([0,1,2],dtype=torch.int32),torch.tensor([0,1],dtype=torch.int32),
                                       torch.tensor([2.,3.]),size=(2,2),device='cuda')
            y=torch.sparse.mm(x,torch.ones((2,2),device='cuda')).cpu()
            result['tiny_sparse_cuda_passed']=bool(torch.equal(y,torch.tensor([[2.,2.],[3.,3.]])))
        else:
            result['tiny_sparse_cuda_passed']=False
        bar.update(1)
        result['scope']='Hardware metadata and a two-nonzero CUDA multiply only; no long job.'
        atomic_json(CODEX/'results/b01_preflight.json',result)
        bar.update(1)
    print(json.dumps(result,indent=2))
    if not result['tiny_sparse_cuda_passed']:
        raise SystemExit('CUDA sparse check failed. Do not start the retrieval pilot yet; share this output.')


if __name__=='__main__':
    main()
