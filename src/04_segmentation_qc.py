from __future__ import annotations
import argparse, re
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from mdp_utils import load_config, setup_logger, write_xlsx, write_csv

MASK_RE=re.compile(r'^(?P<id>.+?)(?P<side>[LR])_(?P<structure>[^.]+)\.nii\.gz$',re.I)
IMAGE_TYPES={'T2','REAL'}

def load_nii(path: Path):
    img=nib.load(str(path)); return img,np.asanyarray(img.dataobj)

def geometry(img):
    return tuple(img.shape),tuple(round(float(x),7) for x in img.header.get_zooms()[:3]),tuple(np.round(img.affine.flatten(),6))

def mask_metrics(path: Path,reference: Path|None) -> dict:
    img,arr=load_nii(path); binary=arr>0; vox=int(binary.sum()); out={'shape':str(img.shape),'spacing':str(img.header.get_zooms()[:3]),'affine':str(np.round(img.affine,6).tolist()),'voxel_count':vox,'empty':vox==0,'geometry_match':None,'reference':str(reference) if reference else ''}
    if reference and reference.exists():
        ref,_=load_nii(reference); out['geometry_match']=geometry(ref)==geometry(img)
    if vox:
        labels,n=ndimage.label(binary,structure=np.ones((3,3,3),dtype=np.uint8)); sizes=np.bincount(labels.ravel())[1:]; out['component_count']=int(n); out['largest_component_fraction']=float(sizes.max()/vox)
        x,y,z=np.where(binary); out['touches_boundary']=bool(x.min()==0 or y.min()==0 or z.min()==0 or x.max()==binary.shape[0]-1 or y.max()==binary.shape[1]-1 or z.max()==binary.shape[2]-1)
    else: out.update(component_count=0,largest_component_fraction=0.0,touches_boundary=False)
    return out

def make_montage(subject: Path,output: Path) -> bool:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes
    t2=next(subject.glob('*_T2.nii.gz'),None)
    if not t2: return False
    base_img=nib.as_closest_canonical(nib.load(str(t2))); base=np.asanyarray(base_img.dataobj).astype(float); lo,hi=np.nanpercentile(base,[1,99]); base=np.clip((base-lo)/(hi-lo+1e-9),0,1)
    masks=[]
    for f in sorted(subject.glob('*.nii.gz')):
        m=MASK_RE.match(f.name)
        if m and m.group('structure').upper() not in IMAGE_TYPES:
            mask_img=nib.as_closest_canonical(nib.load(str(f))); mask=np.asanyarray(mask_img.dataobj)>0
            if mask.shape==base.shape and mask.any(): masks.append((m.group('side').upper(),m.group('structure'),mask))
    if not masks: return False
    fig=plt.figure(figsize=(12,max(3,2.6*len(masks))),dpi=110)
    for row,(side,structure,mask) in enumerate(masks):
        coords=np.where(mask); x,y,z=(int(np.median(c)) for c in coords)
        views=[(base[:,:,z].T,mask[:,:,z].T,'axial'),(base[:,y,:].T,mask[:,y,:].T,'coronal'),(base[x,:,:].T,mask[x,:,:].T,'sagittal')]
        for col,(background,overlay,title) in enumerate(views):
            ax=fig.add_subplot(len(masks),4,row*4+col+1)
            ax.imshow(background,cmap='gray',origin='lower')
            if overlay.any(): ax.contour(overlay.astype(float),levels=[0.5],colors='#D55E00',linewidths=0.8)
            ax.set_title(f'{side} {structure} - {title}',fontsize=8); ax.axis('off')
        ax3=fig.add_subplot(len(masks),4,row*4+4,projection='3d')
        try:
            verts,faces,_,_=marching_cubes(np.pad(mask.astype(np.uint8),1),0.5)
            verts-=1
            if len(faces)>3500: faces=faces[np.linspace(0,len(faces)-1,3500,dtype=int)]
            ax3.plot_trisurf(verts[:,0],verts[:,1],faces,verts[:,2],color='#56B4E9',linewidth=0,alpha=0.95)
            ax3.set_box_aspect(np.maximum(np.ptp(verts,axis=0),1))
        except Exception as e:
            ax3.text2D(0.05,0.5,f'3D failed: {type(e).__name__}',transform=ax3.transAxes,fontsize=7)
        ax3.set_title(f'{side} {structure} - 3D surface',fontsize=8); ax3.set_axis_off()
    fig.suptitle(f'{subject.name}: per-structure segmentation QC',fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.99)); output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,bbox_inches='tight'); plt.close(fig); return True

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--skip-montage',action='store_true'); a=ap.parse_args(); _,p=load_config(a.config); log=setup_logger('segqc',p.logs/'04_segmentation_qc.log'); out=p.output_root/'03_morphometry'; (out/'qc_montage').mkdir(parents=True,exist_ok=True); rows=[]; failures=[]
    for batch in p.segmentation_batches:
        for subject in sorted((p.segmentation_root/batch).glob('sub*')):
            if not subject.is_dir(): continue
            t2=next(subject.glob('*_T2.nii.gz'),None); real=next(subject.glob('*_REAL.nii.gz'),None)
            for f in sorted(subject.glob('*.nii.gz')):
                m=MASK_RE.match(f.name)
                if not m or m.group('structure').upper() in IMAGE_TYPES: continue
                structure=m.group('structure'); ref=real if structure.upper()=='ELS' and real else t2
                try:
                    q=mask_metrics(f,ref); status=[]
                    if q['empty']: status.append('empty')
                    if q['geometry_match'] is False: status.append('geometry_mismatch')
                    if q['component_count']>5 or q['largest_component_fraction']<0.90: status.append('fragmented')
                    if q['touches_boundary']: status.append('touches_boundary')
                    row=[batch,subject.name,m.group('side').upper(),structure,str(f.relative_to(p.project_root)),q['shape'],q['spacing'],q['affine'],q['voxel_count'],q['component_count'],q['largest_component_fraction'],q['touches_boundary'],q['geometry_match'],';'.join(status) if status else 'pass']
                except Exception as e: row=[batch,subject.name,m.group('side').upper(),structure,str(f.relative_to(p.project_root)),'','','',0,0,0,False,False,f'read_error:{type(e).__name__}:{e}']
                rows.append(row)
                if row[-1]!='pass': failures.append(row)
            if not a.skip_montage and batch==p.segmentation_batches[-1]:
                try: make_montage(subject,out/'qc_montage'/f'{batch}_{subject.name}.png')
                except Exception as e: log.warning('montage failed %s: %s',subject,e)
    headers=['batch','seg_subject_id','ear_side','structure','relative_path','shape','spacing','affine','voxel_count','component_count','largest_component_fraction','touches_boundary','geometry_match','qc_status']; passed=sum(r[-1]=='pass' for r in rows)
    write_xlsx(out/'qc_metrics.xlsx',{'qc':(headers,rows),'summary':(['metric','value'],[['mask_files',len(rows)],['pass',passed],['flagged',len(failures)],['pass_rate',passed/len(rows) if rows else None]])}); write_csv(out/'qc_failures.csv',headers,failures); log.info('mask files=%d pass=%d flagged=%d',len(rows),passed,len(failures)); return 0
if __name__=='__main__': raise SystemExit(main())
