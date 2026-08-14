from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class NwmCondition:     #定义输入条件
    dx: float
    dy: float
    dtheta: float
    rel_t: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "dx": float(self.dx),
            "dy": float(self.dy),
            "dtheta": float(self.dtheta),
            "rel_t": float(self.rel_t),
        }


@dataclass
class NwmPrediction:        #定义输出
    pred_latent: Any        #patch特征输出
    pred_rgb: Optional[Any] = None      #解码后的rgb输出
    pred_cls: Optional[Any] = None      #双头预测出的cls / ghost token
    confidence: Optional[Any] = None    #sigmoid后的置信度，范围是0到1
    conf_logit: Optional[Any] = None    #sigmoid前的原始分数
    meta: Dict[str, Any] = field(default_factory=dict)
