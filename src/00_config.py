from __future__ import annotations
import argparse
from pathlib import Path
from mdp_utils import load_config, setup_logger

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); a=p.parse_args()
    _,paths=load_config(a.config); log=setup_logger('config',paths.logs/'00_config.log')
    for name,path in [('PROJECT_ROOT',paths.project_root),('CLINICAL_TABLE',paths.clinical_table),('SEGMENTATION_ROOT',paths.segmentation_root),('PEBM_PAPER',paths.pebm_paper)]:
        if not path.exists(): raise FileNotFoundError(f'{name} not found: {path}')
        log.info('%s=%s',name,path)
    log.info('OUTPUT_ROOT=%s',paths.output_root); return 0
if __name__=='__main__': raise SystemExit(main())

