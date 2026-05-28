"""img_place 与 RP/RC 预测结果到数据库字段的映射。"""

from __future__ import annotations

import json
from typing import Any

# img_place -> (predict_img, primary_class, secondary_class, cross_rack_label)
PLACE_RULES: dict[str, tuple[str, str, str, str]] = {
    "fromrpphoto": ("RP_RACK", "RP", "RC", "RC展架"),
    "fromrcphoto": ("RC_RACK", "RC", "RP", "RP展架"),
}
NON_RACK_FAILURE = "非RP/RC展架"
AUDIT_IMG_PLACES = tuple(PLACE_RULES.keys())


def row_to_prediction_tuple(
    row: tuple[Any, ...],
    result: dict[str, tuple[int, float]],
    model_version: str,
    predict_time: int,
) -> tuple[Any, ...]:
    id_, visit_id, _picture_path, img_place = row

    rule = PLACE_RULES.get(img_place)
    if rule is None:
        return (
            id_,
            visit_id,
            json.dumps(result, ensure_ascii=False),
            None,
            0,
            f"非审核图片: {img_place}",
            model_version,
            predict_time,
        )

    predict_img, primary, secondary, cross_label = rule
    primary_pred = result.get(primary, (0,))[0]
    secondary_pred = result.get(secondary, (0,))[0]

    if primary_pred == 1:
        predict_status = 1
        failure_reason = ""
    elif secondary_pred == 1:
        predict_status = 2
        failure_reason = cross_label
    else:
        predict_status = 3
        failure_reason = NON_RACK_FAILURE

    return (
        id_,
        visit_id,
        json.dumps(result, ensure_ascii=False),
        predict_img,
        predict_status,
        failure_reason,
        model_version,
        predict_time,
    )
