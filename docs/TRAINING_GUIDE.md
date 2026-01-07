# TensorFlow.js 문서 분류기 학습 가이드

## 현재 모델 구조
- **베이스 모델**: MobileNet (경량화 CNN)
- **입력**: 224x224 RGB 이미지
- **출력**: 3개 클래스 (ID_CARD, DRIVER_LICENSE, BUSINESS_REGISTRATION)
- **위치**: `public/models/classifier/model.json`

## 문제점
현재 모델은 3개 클래스만 학습되어 있어, **모든 이미지를 반드시 3개 중 하나로 분류**합니다.
비문서 이미지(만화, 사진 등)도 문서로 분류되는 문제가 있습니다.

## 해결: "OTHER" 클래스 추가 학습

### 1. 학습 데이터 준비

```
training_data/
├── ID_CARD/              # 주민등록증 이미지 (50~100장)
│   ├── id_001.jpg
│   ├── id_002.jpg
│   └── ...
├── DRIVER_LICENSE/       # 운전면허증 이미지 (50~100장)
│   ├── dl_001.jpg
│   └── ...
├── BUSINESS_REGISTRATION/ # 사업자등록증 이미지 (50~100장)
│   ├── br_001.jpg
│   └── ...
└── OTHER/                # 비문서 이미지 (100~200장) ⬅️ 새로 추가
    ├── cartoon_001.jpg
    ├── photo_001.jpg
    ├── screenshot_001.jpg
    └── ...
```

### 2. Python 학습 스크립트 (train_classifier.py)

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflowjs as tfjs

# 설정
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4  # ID_CARD, DRIVER_LICENSE, BUSINESS_REGISTRATION, OTHER

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./127.5,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
    preprocessing_function=lambda x: x - 1  # [-1, 1] 정규화
)

train_generator = train_datagen.flow_from_directory(
    'training_data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'training_data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# MobileNetV2 베이스 모델 (사전 학습 가중치 사용)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 베이스 모델 동결 (Transfer Learning)
base_model.trainable = False

# 분류 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# TensorFlow.js 형식으로 저장
tfjs.converters.save_keras_model(model, 'output_model')

print("모델 저장 완료!")
print("output_model 폴더의 파일들을 public/models/classifier/에 복사하세요.")
```

### 3. 학습 실행

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install tensorflow tensorflowjs pillow

# 학습 실행
python train_classifier.py
```

### 4. 모델 배포

```bash
# 생성된 모델 파일을 프로젝트에 복사
cp output_model/* d:/workspace/ocr/public/models/classifier/
```

### 5. 코드 수정 (tensorflowService.ts)

```typescript
// CLASS_NAMES 배열에 OTHER 추가
const CLASS_NAMES: DocumentType[] = [
    'BUSINESS_REGISTRATION', 
    'DRIVER_LICENSE', 
    'ID_CARD', 
    'OTHER'  // 새로 추가
];
```

```typescript
// types.ts - DocumentType에 OTHER 추가 (이미 UNKNOWN이 있으면 불필요)
export type DocumentType = 
    'BUSINESS_REGISTRATION' | 'ID_CARD' | 'DRIVER_LICENSE' | 'OTHER' | 'UNKNOWN';
```

## 주의사항

1. **클래스 순서**: `flow_from_directory`는 폴더명 알파벳 순으로 클래스를 정렬합니다. 
   - BUSINESS_REGISTRATION (0), DRIVER_LICENSE (1), ID_CARD (2), OTHER (3)
   - `CLASS_NAMES` 배열 순서도 동일하게 맞춰야 합니다.

2. **데이터 균형**: 각 클래스당 비슷한 수의 이미지를 사용하세요.

3. **다양한 OTHER 데이터**: 만화, 사진, 스크린샷, 영수증 등 다양한 비문서 이미지를 포함하세요.
