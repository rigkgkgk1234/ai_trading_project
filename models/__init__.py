"""
models/ — ONNX 모델 로더
─────────────────────────────────────────────────────────
역할:
  1. ONNX 모델 및 스케일러 로드 (시작 시 자동 로드)
  2. model object: {model, scaler, feature_cols, scaler_mean, scaler_scale}
  3. 서버 재시작 시 /model/reload 엔드포인트를 통해 다시 로드
─────────────────────────────────────────────────────────
"""

import joblib
import onnx
import onnxruntime as ort
import numpy as np
from loguru import logger

from config import MODEL_PATH, SCALER_PATH, FEATURE_COLS


# ─────────────────────────────────────────────
# 1. ONNX 모델 로드
# ─────────────────────────────────────────────

def _load_onnx_model(path: str) -> ort.InferenceSession:
    """ONNX 모델을 로드합니다."""
    try:
        session = ort.InferenceSession(path)
        logger.success(f"[Model] ONNX 모델 로드: {path}")
        return session
    except Exception as e:
        logger.warning(f"[Model] ONNX 로딩 경고: {e}")
        # 모델이 아직 없을 수 있음 (처음 시작)
        return None


# ─────────────────────────────────────────────
# 2. 스케일러 로드
# ─────────────────────────────────────────────

def _load_scaler(path: str):
    """SVD 스킨너를 로드합니다."""
    try:
        scaler = joblib.load(path)
        logger.success(f"[Model] 스케일러 로드: {path}")
        return scaler
    except Exception as e:
        logger.warning(f"[Model] 스케일러 로딩 경고: {e}")
        return None


# ─────────────────────────────────────────────
# 3. 모델 객체 생성
# ─────────────────────────────────────────────

class ModelContainer:
    """모델을 래핑한 컨테이너입니다."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = FEATURE_COLS
        self.scaler_mean = None
        self.scaler_scale = None
        self._loaded = False

    def reload(self) -> bool:
        """
        모델과 스케일러를 새로 로드합니다.
        서버 시작 시 자동으로 호출됩니다.
        """
        if self._loaded:
            logger.info("[Model] 이미 로드된 모델이 있습니다.")
            return True

        logger.info("[Model] 시작 시 모델 로드 시작...")

        # ONNX 모델 로드
        session = _load_onnx_model(MODEL_PATH)
        if session is None:
            logger.warning(
                "[Model] ONNX 모델 없음 - 학습된 모델이 없습니다. "
                "(models/train_model.py 를 먼저 실행하세요)"
            )
            # 빈 ONNX 세션 생성 (추후 로딩용)
            self._create_dummy_session()
            self._loaded = True
            return False

        # 스케일러 로드
        scaler = _load_scaler(SCALER_PATH)
        if scaler:
            self.scaler_mean = scaler.mean_
            self.scaler_scale = scaler.scale_

        self.model = session
        self._loaded = True

        logger.success("[Model] 시작 시 모델 로드 완료!")
        return True

    def _create_dummy_session(self):
        """
        모델이 없을 때 빈 ONNX 세션 생성 (시그니처만).
        추후 학습된 모델을 로드했을 때 교체됩니다.
        """
        # 빈 모델 생성 (XGBoost multi-class 예시)
        dummy_model = onnx.ModelProto()
        dummy_inputs = dummy_model.graph.input
        for i, col in enumerate(self.feature_cols):
            inp = dummy_inputs.add()
            inp.name = f"float_input_{i}"
            inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        dummy_outputs = dummy_model.graph.output
        for i, cls in enumerate(["HOLD", "BUY", "SELL"]):
            out = dummy_outputs.add()
            out.name = f"output_{i}"
            out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        dummy_node = dummy_model.graph.node.add()
        dummy_node.op_type = "Identity"
        dummy_node.input.append("float_input_0")
        dummy_node.output.append("output_0")

        dummy_model.graph.input.extend(dummy_model.graph.input[1:])  # keep first
        # dummy_model.graph.output.extend([dummy_outputs[0]])

        import onnx
        dummy_model = onnx.ModelProto()
        # 더 간단히: dummy input/output만
        pass

        self.model = ort.InferenceSession(dummy_model.SerializeToString())
        self._loaded = True


# ─────────────────────────────────────────────
# 4. 글로벌 모델 인스턴스
# ─────────────────────────────────────────────

_model = ModelContainer()


def get_model() -> ModelContainer:
    """글로벌 모델 인스턴스를 반환합니다."""
    return _model


# ─────────────────────────────────────────────
# 5. 서버 시작 시 자동 로드 (lazy loading)
# ─────────────────────────────────────────────

def load_model_on_startup():
    """서버 시작 시 모델 로드."""
    get_model().reload()


# ─────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = get_model()
    model.reload()
    print(f"모델 로드됨: {model._loaded}")
    print(f"피처: {model.feature_cols}")
    print(f"종류: {type(model.model)}")
