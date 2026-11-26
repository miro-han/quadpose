# Phase IV · 동일 물리 모서리 검증과 자세 재추정

한 시점에서 상자 모서리가 손 뒤로 사라졌을 때, 다른 카메라의 검출 선분을 같은 모델 모서리에 연결할 수 있을까? 대응 검사를 강화하면 잘못된 대응과 유용한 대응이 모두 줄 수 있다. 이번 실험은 각 방법의 이전 출력에서 다시 시작해 이 선택이 이후 자세·가림·재획득에 미치는 영향을 측정한다.

- [알고리즘과 수식](ALGORITHM_DESIGN.md) · [구현](../../pose/refinement.py) · [네 개의 3시점 가설 및 선택](../../pose/hypotheses.py) · [복구 정책](../../pose/recovery.py)
- [본 시험 결과와 원시 자료](ABLATION_STUDY.md) · [단계별 효과](ABLATION_STUDY.md) · [실패 사례](FAILURE_ANALYSIS.md) · [참고 문헌](REFERENCES.md)
- [고정 실험 설정](benchmark_config.json) · [합성 데이터와 GT 경계](benchmark_data.py) · [측정과 집계 코드](benchmark.py) · [팀 기술 안내서](../../docs/guide/INDEX.md)

## 재현

프로젝트 루트에서 기존 Windows Python 3.12 환경을 사용한다. 아래는 PowerShell 명령이다. 계산은 CPU에서 실행한다. 출력 폴더는 새 이름을 사용한다. 전체 비교는 다섯 suite를 순차 실행한다.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$py = '.\.venv\Scripts\python.exe'
& $py -B -m unittest discover -s evaluation/pose_benchmark -t . -p 'test_*.py'
& $py -B -m unittest discover -s pose -t . -p 'test_*.py'
foreach ($suite in 'selection','assignment','primary','recovery','observability') {
    & $py -B -m evaluation.resource_limits --threads 1 --memory-gib 3 evaluation.pose_benchmark.benchmark --suite $suite --output "evaluation/pose_benchmark/reproduction/$suite"
    if ($LASTEXITCODE -ne 0) { throw "Failed $suite" }
}
& $py -B -m evaluation.pose_benchmark.verify_preservation
```

빠른 연결 확인은 한 suite에 `--smoke`와 새 출력 경로를 준다. smoke 출력은 최종 결과에 포함하지 않는다. manifest에서 시드·설정·구현 해시·최고 메모리를 확인할 수 있다. 측정 시간은 하드웨어·CPU 부하에 따라 달라진다. 고정 자극과 시드로 얻은 오차와 유효율, 관측의 분모를 먼저 대조한다.

실물에서 결론을 내려면 보정한 4대 카메라의 동시 녹화, 수작업 물리 ID 정답과 독립 자세 기준이 필요하다. 실험 후보를 제품 기본값으로 바꾸는 결정은 [현장 검증](../../docs/guide/15_PHYSICAL_VALIDATION.md)에 남겼다.
