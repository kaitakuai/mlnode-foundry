"""Decode-PoC pass-through for MLNode's pow_v2 routes (in-image form).

Stage-4 mirror of kaitakuai/gonka@feat/decode-poc-mlnode commits 86ff63d + c258ab3
("feat(mlnode): decode-PoC pass-through in pow_v2 routes"). The base image
carries the release mlnode whose API layer silently filters everything
decode: PoCParamsModel rejects max_tokens/route_window (extra=forbid -> 422)
and ArtifactModel strips k_points_steps from validation payloads, so the
validator would teacher-force against nothing.

Anchored textual edits, additive only; any missing anchor fails the build
(= upstream refactored the file, re-verify against the fork commit).
"""
import sys

FILE = "/app/packages/api/src/api/inference/pow_v2_routes.py"

EDITS = [
    # 1) decode params reach the backend
    ('''class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12''',
     '''class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12
    # Decode-PoC (kaitakuai/gonka@feat/decode-poc-mlnode 86ff63d): max_tokens
    # > 0 selects the decode scheme; route_window is the consensus routing
    # window (release value 256).
    max_tokens: int = 0
    route_window: int = 256'''),
    # 2) the minimal decode artifact: the k-id chain
    ('''class ArtifactModel(BaseModel):
    nonce: int
    vector_b64: str''',
     '''class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nonce: int
    vector_b64: str = ""
    # Decode artifact payload: the k-id chain, also the teacher-forcing
    # reference during validation. Without this field the router silently
    # strips trajectories from validation.artifacts.
    k_points_steps: Optional[List[int]] = None'''),
    # 3) decode verdict defaults (agreed 2026-08-09): pooled binomial over the
    #    200-artifact sample at tau=0 with p_mismatch=0.1198
    ('''class StatTestModel(BaseModel):
    dist_threshold: float = 0.02
    p_mismatch: float = 0.001
    fraud_threshold: float = 0.01''',
     '''class StatTestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Prefill knob; ignored by the decode backend (wire compat).
    dist_threshold: float = 0.02
    # Decode verdict: backend pools all steps of the validation batch and
    # runs a binomial test at margin gate tau=0 (backend default) with this
    # baseline. 0.1198 sits between measured honest (9.92%) and fraud
    # (13.91%) shares on the worst pair (A100 prover -> B300 validator).
    p_mismatch: float = 0.1198
    fraud_threshold: float = 0.01'''),
    # 4) batch_size: backend default (image profile) decides the round batch
    ('''    node_count: int
    batch_size: int = 32
    params: PoCParamsModel
    url: Optional[str] = None
    poc_stronger_rng: bool = False''',
     '''    node_count: int
    # None => omitted from the forwarded payload; the backend default
    # (POC_BATCH_SIZE_DEFAULT, owned by the image profile) decides the decode
    # round batch — mlnode cannot know the card's KV wall.
    batch_size: Optional[int] = None
    params: PoCParamsModel
    url: Optional[str] = None
    poc_stronger_rng: bool = False'''),
    ('''    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = 32
    wait: bool = False''',
     '''    nonces: List[int]
    params: PoCParamsModel
    batch_size: Optional[int] = None
    wait: bool = False'''),
    # 4b) teacher-forcing + loud 422 on unknown fields (fork commit c258ab3;
    #     the silent enforced_k_steps drop was hardware-confirmed 2026-08-17)
    ('''    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None
    poc_stronger_rng: bool = False''',
     '''    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None
    # Teacher-forcing mode: {nonce: k-trajectory}; the backend flips into
    # validation when present. Silently dropped before this field existed
    # here (hardware-confirmed 2026-08-17).
    enforced_k_steps: Optional[Dict[int, List[int]]] = None
    poc_stronger_rng: bool = False'''),
    ('class PoCGenerateRequest(BaseModel):\n    """Request for /generate endpoint."""',
     'class PoCGenerateRequest(BaseModel):\n    """Request for /generate endpoint."""\n    model_config = ConfigDict(extra="forbid")'),
    ('class PoCInitGenerateRequest(BaseModel):\n    """MLNode /init/generate request - group_id/n_groups omitted (injected by MLNode)."""',
     'class PoCInitGenerateRequest(BaseModel):\n    """MLNode /init/generate request - group_id/n_groups omitted (injected by MLNode)."""\n    model_config = ConfigDict(extra="forbid")'),
    ('class ValidationModel(BaseModel):\n    artifacts: List[ArtifactModel]',
     'class ValidationModel(BaseModel):\n    model_config = ConfigDict(extra="forbid")\n    artifacts: List[ArtifactModel]'),
    ('from typing import List, Optional',
     'from typing import Dict, List, Optional'),
    # 5) forward without None fields
    ('''        payload = body.model_dump()
        payload["group_id"] = group_id''',
     '''        payload = body.model_dump(exclude_none=True)
        payload["group_id"] = group_id'''),
    ('r = await call_backend(port, "POST", "/api/v1/pow/generate", body.model_dump())',
     'r = await call_backend(port, "POST", "/api/v1/pow/generate", body.model_dump(exclude_none=True))'),
]


def main() -> int:
    with open(FILE) as f:
        src = f.read()

    if "k_points_steps" in src:
        print("pow-v2-decode-fields: already applied (base caught up); no-op")
        return 0

    for i, (old, new) in enumerate(EDITS, 1):
        if old not in src:
            sys.stderr.write(
                f"ERROR: pow-v2-decode-fields: anchor #{i} not found in {FILE} "
                "— upstream refactored pow_v2_routes.py, re-verify against "
                "kaitakuai/gonka@feat/decode-poc-mlnode 86ff63d\n")
            return 1
        src = src.replace(old, new, 1)

    with open(FILE, "w") as f:
        f.write(src)
    import ast
    ast.parse(open(FILE).read())
    print("pow-v2-decode-fields: decode pass-through applied to pow_v2_routes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
