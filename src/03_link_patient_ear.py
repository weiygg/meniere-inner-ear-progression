from __future__ import annotations
import argparse, re
from collections import defaultdict
from pathlib import Path
from mdp_utils import load_config, setup_logger, read_ear_records, sha256, write_xlsx

PAT=re.compile(r'^sub(?P<id>[^\\/]+)$',re.I)
FILE=re.compile(r'^(?P<id>[^LR_]+)(?P<side>[LR])_(?P<structure>.+)\.nii\.gz$',re.I)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); a=ap.parse_args()
    _,p=load_config(a.config); log=setup_logger('linkage',p.logs/'03_link_patient_ear.log')
    clinical=read_ear_records(p.clinical_table); baseline=[r for r in clinical if not r['is_followup']]
    num_map=defaultdict(set)
    for r in baseline:
        numeric=re.match(r'^(\d+)',r['source_subject_id'])
        if numeric: num_map[f'{int(numeric.group(1)):03d}'].add(r['patient_id'])
    seg=[]
    for batch in p.segmentation_batches:
        bdir=p.segmentation_root/batch
        for subject in bdir.glob('sub*'):
            if not subject.is_dir(): continue
            sid=subject.name[3:]
            candidates=sorted(num_map.get(sid,set()))
            status='matched' if len(candidates)==1 else 'ambiguous' if len(candidates)>1 else 'unmatched'
            for f in subject.glob('*.nii.gz'):
                m=FILE.match(f.name)
                seg.append([batch,sid,m.group('side').upper() if m else '',m.group('structure') if m else f.name,str(f.relative_to(p.project_root)),status,';'.join(candidates)])
    overlaps=[[sid,';'.join(sorted(ids)),len(ids)] for sid,ids in sorted(num_map.items()) if len(ids)>1]
    batch_dups=defaultdict(list)
    for row in seg: batch_dups[(row[1],row[2],row[3])].append(row[0])
    duprows=[[k[0],k[1],k[2],';'.join(v),len(v)] for k,v in batch_dups.items() if len(v)>1]
    out=p.output_root/'01_data_audit'; headers=['segmentation_batch','seg_subject_id','ear_side','structure','relative_path','link_status','candidate_patient_ids']
    clinical_keys=[[r['patient_id'],r['ear_side'],r['ear_id'],r['visit_id'],r['source_site'],r['source_subject_id']] for r in baseline]
    write_xlsx(out/'patient_ear_linkage.xlsx',{
        'clinical_patient_ear_keys':(['patient_id','ear_side','ear_id','visit_id','source_site','source_subject_id'],clinical_keys),
        'segmentation_linkage':(headers,seg),
        'ambiguous_numeric_ids':(['numeric_id','candidate_patient_ids','candidate_n'],overlaps),
        'cross_batch_duplicates':(['seg_subject_id','ear_side','structure','batches','batch_n'],duprows),
    })
    issues=[
        '# NEED CONFIRMATION','',
        'Formal clinical P-EBM and clinical-imaging fusion are stopped until the blocking items below are resolved.','',
        '## 1. Segmentation subject ID is not unique across clinical sources','',
        f'- Problem: {len(overlaps)} numeric IDs occur in both `丽水` and `浙二`, but refer to different patients; segmentation folders use only `subNNN`.',
        '- Files/fields: `MD患者评估20260713.xlsx` (`丽水.ID`, `浙二.ID`) and all segmentation batch folder names.',
        '- Current inference: numeric matching alone is unsafe.',
        '- Optional interpretations: provide a center-specific ID map, rename a copied linkage manifest with source prefixes, or identify which clinical sheet each segmentation batch belongs to.',
        '- Impact: patient-ear linkage, index-ear selection, imaging-clinical fusion and real-data pilot P-EBM may be wrong.',
        '- Recommended choice: provide a two-column mapping `seg_subject_id -> source_site/source_subject_id` plus ear-side confirmation.','',
        '## 2. Authoritative segmentation batch is unclear','',
        f'- Problem: `{p.segmentation_batches[0]}`, `seg3`, and `seg4` contain overlapping subject/ear/structure files; their explanation documents are identical.',
        '- Current inference: `seg4` is newest but is not assumed to be final for modeling.',
        '- Impact: duplicate measurements and version mixing.',
        '- Recommended choice: confirm the authoritative batch or provide a per-subject version manifest.','',
        '## 3. Structure naming and label semantics require confirmation','',
        '- Problem: cochlear masks occur under both `Chochlear` and `Cholear`; current files are separate binary NIfTI masks rather than one documented label map.',
        '- Current inference: these look like spelling variants, but they are not silently merged.',
        '- Impact: structure-level aggregation and anatomical endotype scores can split the same anatomy across names.',
        '- Recommended choice: approve a canonical structure mapping and identify whether `TV` and `ELS` are the intended anatomical structures.','',
        '## 4. Clinical coding definitions require confirmation','',
        '- Problem: direction and exact meaning of CochEH, VestEH, VA, ES/ED, sex codes, ear fullness and AAO-HNS stage are not fully documented in the workbook.',
        '- Impact: cannot define event likelihoods, abnormality direction or independent validation outcomes safely.',
        '- Recommended choice: supply a coding dictionary, including clinical thresholds and whether higher values always mean worse abnormality.','',
        '## 5. The patient-level `浙二MD` sheet is not linked to the ear-level `浙二` sheet','',
        '- Problem: `浙二MD` contains scan date, onset time and additional questionnaires, but uses `编号` rather than the confirmed ear-level `ID + side` key.',
        '- Current inference: no cross-sheet join is attempted because identifier equivalence is undocumented.',
        '- Impact: examination intervals, treatment/symptom covariates and some potential independent outcomes cannot enter the current cohort.',
        '- Recommended choice: provide or confirm a `浙二MD.编号 -> 浙二.ID` mapping and its visit-date semantics.','',
        '## 6. Index ear / bilateral status is not uniquely encoded for all patients','',
        '- Problem: an AAO-HNS stage is present for some ear rows, but this alone does not establish unilateral/bilateral MD or first-onset ear.',
        '- Impact: primary patient-level cohort is incomplete.',
        '- Recommended choice: provide affected side, first-onset side and bilateral status fields.','',
        '## 7. P-EBM event-state reference groups are not yet defined','',
        '- Problem: controls/pathologic labels required for a priori event PDFs are not unambiguously encoded for every candidate biomarker.',
        '- Impact: mixture/event likelihood fitting is not defensible.',
        '- Recommended choice: confirm healthy/disease-control definitions and clinical abnormal thresholds.','',
        '## 8. True mirrored left-right surface asymmetry is not yet defensible','',
        '- Problem: scanner/world orientation and a subject-level midsagittal reflection plane are not documented.',
        '- Current output: intrinsic left-right relative differences are reported for volume, area and maximum diameter only.',
        '- Impact: a pointwise mirrored surface asymmetry index could be geometrically wrong.',
        '- Recommended choice: confirm a common orientation/registration protocol before surface reflection analysis.',''
    ]
    (p.output_root/'NEED_CONFIRMATION.md').write_text('\n'.join(issues),encoding='utf-8')
    log.info('seg files=%d ambiguous numeric ids=%d duplicate structures=%d',len(seg),len(overlaps),len(duprows)); return 0
if __name__=='__main__': raise SystemExit(main())
