import SimpleITK as sitk
from conftest import load_script

def test_physical_volume_uses_spacing(tmp_path):
    mod=load_script('05_extract_inner_ear_morphometry.py')
    img=sitk.Image([10,10,10],sitk.sitkUInt8); img.SetSpacing((0.5,0.5,2.0)); img[2:6,2:6,2:6]=1
    p=tmp_path/'mask.nii.gz'; sitk.WriteImage(img,str(p)); f=mod.features(p)
    assert abs(f['volume_mm3']-32.0)<1e-6
    assert f['surface_area_mm2']>0

