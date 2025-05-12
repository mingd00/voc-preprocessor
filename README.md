# VOC Preprocessor
시각 지능 딥러닝 학습을 위해 VOC2017 데이터를 전처리

## 주요 기능
- 클래스 이름을 기반으로 인덱스를 매핑하고 JSON으로 저장할 수 있다.
- 학습, 검증, 테스트 데이터셋으로 분할할 수 있다.
- 바운딩 박스와 클래스 정보를 학습 가능한 형식으로 변환할 수 있다.
- 변환된 데이터를 이미지로 시각화하여 확인할 수 있다.
- VOC 데이터셋 구조를 점검하거나 압축을 해제할 수 있다.

## 파일 설명

- `matching_idx_class.py`: VOC 메타데이터를 기반으로 클래스 인덱스를 생성
- `data_split.py`: 전체 데이터를 학습/검증/테스트로 분할
- `prepare_target.py`: 바운딩 박스와 클래스 정보를 학습에 적합한 형태로 변환
- `show_transformed_bbox.py`: 변환된 바운딩 박스를 이미지에 시각화
- `check_voc.py`: VOC 데이터셋 구조의 이상 여부를 점검
- `unzip_voc.py`: VOC 압축 파일을 자동으로 해제
- `config.py`: 경로 및 설정 정보를 정의

## 설치 방법

```bash
pip install -r requirements.txt
