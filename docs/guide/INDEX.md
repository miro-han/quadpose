# Team 7 기술 가이드

실패가 생기는 원리를 먼저 보고 싶다면 [가짜 모서리, 고장 난 카메라, 재획득의 세 장면](12_LIMITATIONS.md)을 읽는다.

손에 든 상자의 앞 모서리가 한 화면에서 사라질 때, 다른 영상에서 같은 모서리를 찾고 상자의 자세를 계속 측정할 수 있을까? 이 가이드는 그 질문을 영상 관측, 기하학적 가설, 실패를 포함한 검증의 순서로 따라간다.

| 읽는 목적 | 권장 경로 |
|---|---|
| Quick Overview · 회의 전에 | [팀 요약과 요구사항](01_PROJECT_OVERVIEW.md) → [시스템 그림](03_SYSTEM_ARCHITECTURE.md) → [실험 해석](10_EXPERIMENTAL_RESULTS.md) → [현장 검증](15_PHYSICAL_VALIDATION.md) |
| Mathematical Study · 수식을 설명하려면 | [좌표와 보정](04_CAMERA_GEOMETRY.md) → [대응과 삼각측량](06_FEATURE_CORRESPONDENCE.md) → [강체 적합](07_POSE_ESTIMATION.md) → [부분집합 평가](08_MULTI_VIEW_FUSION.md) |
| Implementation & Reproduction · 실험을 인수하면 | [입력 영상](05_IMAGE_PROCESSING.md) → [실행 순서](13_IMPLEMENTATION_GUIDE.md) → [IV 설계](../../evaluation/pose_benchmark/ALGORITHM_DESIGN.md) → [원시 결과로 가는 길](../../evaluation/pose_benchmark/ABLATION_STUDY.md) → [성능 계측](11_REALTIME_PERFORMANCE.md) |

설치 후보는 [카메라 배치](09_CAMERA_LAYOUT.md), 검증 한계와 남은 선택은 [현장 검증](15_PHYSICAL_VALIDATION.md), 채택 근거와 당시 판단은 [개발 기록](14_DEVELOPMENT_HISTORY.md)에서 찾을 수 있다. [용어집](GLOSSARY.md) · [참고 문헌](REFERENCES.md) · [기존 문서 목록](SOURCE_MAP.md)
