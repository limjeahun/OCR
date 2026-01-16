# OCR 문서 인식 시스템 (Frontend)

![Next.js](https://img.shields.io/badge/Next.js-16.0.10-000000?style=flat&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19.2.1-61DAFB?style=flat&logo=react&logoColor=black)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat&logo=typescript&logoColor=white)

사업자등록증, 주민등록증, 운전면허증을 촬영하면 자동으로 텍스트를 추출해주는 웹 애플리케이션입니다.


## 주요 기능

- **문서 분류**: TensorFlow.js로 업로드된 이미지가 어떤 문서인지 판별
- **품질 체크**: 해상도, 밝기, 대비를 분석해서 OCR 성공률 예측
- **결과 수정**: 추출된 데이터를 사용자가 직접 확인하고 수정 가능

## 기술 스택

- Next.js 16 / React 19 / TypeScript
- TensorFlow.js (브라우저 내 문서 분류)
- OpenCV.js (이미지 전처리)

## 시스템 구성

```
[Frontend] → [API Server] → [Kafka] → [Worker]
                                         ↓
                                    [OCR 엔진들]
                                    (PaddleOCR, Pororo, EasyOCR)
                                         ↓
                                    [Gemma3 LLM]
                                    (결과 교차 검증)
```

Frontend에서 이미지를 업로드하면:
1. 브라우저에서 TensorFlow.js로 문서 유형 분류 (신뢰도 70% 미만이면 거부)
2. 품질 분석 후 API 서버로 전송
3. Kafka를 통해 Worker로 비동기 처리
4. 3개 OCR 엔진 병렬 실행 → LLM으로 결과 취합
5. 사용자가 결과 확인 후 저장

## 실행 방법

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# http://localhost:3000 접속
```

Backend와 OCR 엔진은 [Backend 저장소](https://github.com/limjeahun/Merchant-Management-System)의 가이드를 참고하세요.

## 프로젝트 구조

```
src/
├── app/                    # Next.js App Router
├── components/
│   ├── OCRScanner.tsx     # 메인 스캔 화면
│   └── BusinessRegistrationForm.tsx
└── services/
    ├── ocr/
    │   └── tensorflowService.ts  # 문서 분류, 품질 분석
    └── api/
        └── ocrApi.ts             # Backend 통신
```

## 지원 문서

| 문서 | 추출 항목 |
|------|----------|
| 사업자등록증 (개인) | 상호, 사업자번호, 대표자명, 주소, 업태, 종목, 개업일 |
| 사업자등록증 (법인) | 상호, 사업자번호, 법인등록번호, 대표자명, 소재지, 업태, 종목 |
| 주민등록증 | 성명, 주민등록번호(마스킹), 주소, 발급일 |
| 운전면허증 | 성명, 면허번호, 면허종류, 주소, 발급일 |

## 왜 이렇게 만들었나

**단일 OCR 엔진의 한계**

처음엔 PaddleOCR 하나만 썼는데, 흐릿하거나 기울어진 문서에서 인식률이 너무 낮았습니다.
그래서 3개 엔진을 병렬로 돌리고 LLM이 결과를 비교해서 제일 그럴듯한 값을 뽑도록 했습니다.

**비문서 이미지 처리**

만화나 일반 사진을 올려도 OCR을 시도하는 문제가 있어서,
Frontend에서 먼저 문서인지 판별하고, Backend에서도 한 번 더 품질 검사합니다.

**타임아웃 문제**

OCR이 오래 걸리면 API가 타임아웃 나서 Kafka로 비동기 처리하도록 변경했습니다.
요청하면 바로 requestId 받고, 결과는 폴링으로 가져옵니다.
