# 실제 단일 영상의 관측 후보 처리 시간

동봉 1920×1080 영상의 원본 0–29번 프레임, 보관된 SAM 마스크, 같은 프레임의 LSD 선분 로그를 입력으로 사용했다. CPU와 AMD UMat 양쪽에서 가우시안 블러와 Canny 15/45를 적용하고 5×5 팽창 마스크로 에지를 제한했다. AMD 시간에는 프레임·마스크 업로드, 선택 에지의 호스트 복사와 완료 동기화가 포함된다. 이후 ROI·에지 지지율 필터와 유한 선분 교차점 추출은 두 경로 모두 같은 CPU 코드에서 처리한다. SAM 추론과 LSD 계산 시간은 포함하지 않았다.

| 30프레임 클립 | CPU p50 / p95 | AMD UMat 경로 p50 / p95 |
| --- | ---: | ---: |
| 전체, 동일한 영상·PNG·JSONL 읽기 포함 | 3902.22 / 3985.33 ms | 3650.74 / 3750.91 ms |
| 호스트 메모리에 입력을 읽어 둔 상태 | 2568.24 / 2593.80 ms | 2368.51 / 2408.51 ms |
| 선택 Canny 단계, 호스트 메모리에 입력을 읽어 둔 상태 | 546.02 / 551.64 ms | 324.82 / 337.82 ms |

선택 에지 단계의 중앙값은 40.5%, 전체 클립 중앙값은 6.4% 감소했다. 필터와 교차점 추출이 양쪽에서 약 2.0초를 차지했다. 같은 장면에서 두 경로의 에지 전경 IoU는 99.9692%였으며, CPU 에지는 보관된 에지 PNG 30개와 전부 일치했다. 수용한 선분 ID 및 거절 이유는 모두 같았다. 양쪽이 생성한 교차점 후보 134개의 좌표도 모두 같았다. 이는 출력 일치도이며 정답 물리 모서리·꼭짓점 정확도는 측정하지 않았다. [측정 JSON](measurement.json)

OpenCV 5.0.0에서 CPU 1스레드와 3 GiB 프로세스 커밋 상한을 적용했다. 각 경로를 한 클립으로 워밍업한 뒤 동일 클립을 다섯 번 반복했고, 반복마다 실행 순서를 바꿨다. 운영체제 파일 캐시와 공유 호스트 부하는 통제하지 않았다. AMD DriverStore 런타임을 직접 지정하고 `AMD:GPU:gfx1035`를 확인했다. UMat은 일부 연산을 CPU로 처리할 수 있으며 이 모듈은 커널별 엔진 사용률을 수집하지 않는다. 물리 카메라 여러 대의 캡처·정렬·융합 지연은 이 실험에 포함되지 않는다.

```powershell
python -B -m experiments.amd.opencl.observations.benchmark --output experiments/amd/opencl/observations/replay.json
python -B -m experiments.amd.opencl.observations.benchmark --cpu-only --repeats 3 --output experiments/amd/opencl/observations/cpu_replay.json
```

AMD 선택이 가능하지 않으면 CPU만 측정하고 `mode: "cpu_only"`와 장치 상태 이유를 기록한다. 강제로 CPU만 사용한 별도 3회 실행에서 독립 CPU 재실행의 결과가 일치했다. [CPU 경로 검증 JSON](cpu_fallback.json)
