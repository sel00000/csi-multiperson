# CSI Multiperson

ESP32 TX3×RX3 WiFi CSI를 이용한 다중 인원 자세 추정을 위한 Python 패키지입니다.

![CSI 기반 다중 인원 자세 추정 아키텍처](docs/multiperson-csi-design/figures/multiperson-csi-architecture.png)

위 그림은 전체 설계안입니다. 현재 S00–S07/V00 범위의 설정·파서·epoch 판정·시계 변환·세션 저장·causal grid/window·특징 생성·계측 및 화면 출력 계약을 구현하고 합성 fixture로 검증했습니다. 그림의 모델 학습·다중 인원 자세 추론과 실장치 검증은 후속 단계입니다.

## 코드와 검증

- [실행 방법과 API](csi-multiperson/README.md)
- [소스 코드](csi-multiperson/src/csi_mp/)
- [테스트](csi-multiperson/tests/)
- [검증 보고서](csi-multiperson/reports/)
- [아키텍처 그림 원본](docs/multiperson-csi-design/figures/multiperson-csi-architecture.png)

코드·설정·fixture·검증 보고서는 `csi-multiperson/`에, 그림은 `docs/multiperson-csi-design/figures/`에 있습니다.
