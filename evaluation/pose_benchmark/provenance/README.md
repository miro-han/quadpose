# 작업 시작 상태와 보존 확인

프로젝트 루트와 상위 작업 폴더에 Git 저장소가 없어 커밋 대신 SHA-256 목록을 기준으로 보존 상태를 확인했다. [원본 전달 목록](../../FULL_PACKAGE_MANIFEST.json)은 인계 시점의 831개 파일을 포함한다. [별도 시작 목록](before.json)은 Phase I–III 연구 코드·실험 산출물·문서, 매니페스트, README, 두 ZIP을 포함한 642개 항목이다. 두 목록은 겹치며 설치된 가상환경 전체에 대한 해시 목록은 아니다.

루트 README 개정 전 [원본 바이트 사본](README.before.md)을 저장했다. 원래 README에 있는 링크는 원래 프로젝트 루트 기준이다. 검증 도중 원래 경로에서 사라진 두 ZIP은 휴지통에서 시작 해시가 같은 파일을 확인하고 기존 경로로 복사했다. SHA-256은 인계 ZIP `ca0f02b05bde9227cd1f29b612b7e827ea49142baa097dfb8909a51dfa8ff87a`, Phase II ZIP `2b3afd2fac94de349888993188b3cdf6a6e526a69d233e714b919e2fbf508790`이다. 검증 시 원래 README 항목은 보존 사본과 비교한다.

```text
FULL_PACKAGE_MANIFEST.json: 831/831 verified
phase4/provenance/before.json: 642/642 verified
```

[검증 프로그램](../verify_preservation.py) · [단계별 미해결 사항 및 결정 이력](../../docs/guide/14_DEVELOPMENT_HISTORY.md) · [현장 미검증 항목](../../docs/guide/15_PHYSICAL_VALIDATION.md)
