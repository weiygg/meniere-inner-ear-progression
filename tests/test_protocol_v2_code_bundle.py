from scripts.build_protocol_v2_code_bundle import is_allowed_member, sha256_bytes


def test_code_bundle_allows_reproducibility_sources() -> None:
    for path in (
        "scripts/run_nnunet_protocol_v2.py",
        "src/meniere_progression/segmentation/nnunet_trainers.py",
        "configs/segmentation_experiments.yaml",
        "tests/test_protocol_v2_nnunet_trainers.py",
        "docs/PROTOCOL_V2_EXECUTION_RUNBOOK.md",
        "requirements-segmentation.txt",
    ):
        assert is_allowed_member(path) == (True, "allowed")


def test_code_bundle_rejects_protected_data_and_weights() -> None:
    expected = {
        "data/patient_table.csv": "protected_file_type",
        "results/checkpoint_final.pth": "protected_file_type",
        "seg4/sub128/138R_HSC.nii.gz": "protected_file_type",
        "data/manifests/clinical_feature_schema.xlsx": "protected_file_type",
    }
    for path, reason in expected.items():
        assert is_allowed_member(path) == (False, reason)


def test_code_bundle_rejects_unrelated_tracked_content() -> None:
    assert is_allowed_member("misc/file.bin") == (False, "outside_code_allowlist")


def test_sha256_bytes_is_stable() -> None:
    assert sha256_bytes(b"Protocol V2\n") == "6b45688a18e523570a6dcaa8b41254814617036b63b9be929d56baf5a4e0db53"
