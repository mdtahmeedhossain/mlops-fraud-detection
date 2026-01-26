import json
import logging
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import DRIFT_THRESHOLD, REFERENCE_DATA_FILE
from src.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitors data drift between reference (training) and current (production) data.

    Uses the same preprocessing pipeline as training to ensure fair comparison.
    """

    def __init__(self):
        self.reference_data = None
        self.preprocessor: FeaturePreprocessor | None = None
        self._loaded = False

    def load_reference(self) -> None:
        if not REFERENCE_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Reference data not found at {REFERENCE_DATA_FILE}. "
                "Run training first."
            )
        self.reference_data = pd.read_parquet(REFERENCE_DATA_FILE)

        # Load preprocessor to transform incoming raw data
        self.preprocessor = FeaturePreprocessor()
        self.preprocessor.load()

        self._loaded = True
        logger.info(
            f"Reference data loaded: {self.reference_data.shape[0]} rows, "
            f"{self.reference_data.shape[1]} features"
        )

    def check_drift(
        self, current_data: pd.DataFrame, is_preprocessed: bool = False
    ) -> dict:
        """
        Check for data drift between reference and current data.

        Args:
            current_data: The current production data to compare
            is_preprocessed: If True, skip preprocessing (data is already transformed).
                           If False (default), apply preprocessing first.
        """
        if not self._loaded:
            self.load_reference()

        # Preprocess current data if needed (for raw API inputs)
        if not is_preprocessed:
            curr = self.preprocessor.transform(current_data)
        else:
            curr = current_data

        # Use all reference columns for comparison
        ref = self.reference_data

        # Align columns - use intersection to handle any column mismatches
        common_cols = list(set(ref.columns) & set(curr.columns))
        if len(common_cols) < len(ref.columns):
            missing = set(ref.columns) - set(curr.columns)
            logger.warning(f"Missing columns in current data: {missing}")

        ref_aligned = ref[common_cols]
        curr_aligned = curr[common_cols]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_aligned, current_data=curr_aligned)
        report_dict = json.loads(report.json())

        metrics = report_dict.get("metrics", [])
        drift_result = {
            "n_features_analyzed": len(common_cols),
            "n_reference_rows": len(ref_aligned),
            "n_current_rows": len(curr_aligned),
            "drift_detected": False,
            "n_drifted_features": 0,
            "drifted_features": [],
            "drift_share": 0.0,
        }

        for metric in metrics:
            metric_result = metric.get("result", {})
            if "number_of_drifted_columns" in metric_result:
                drift_result["n_drifted_features"] = metric_result[
                    "number_of_drifted_columns"
                ]
                drift_result["drift_share"] = metric_result.get(
                    "share_of_drifted_columns", 0.0
                )
                drift_result["drift_detected"] = (
                    drift_result["drift_share"] > DRIFT_THRESHOLD
                )

                drift_by_columns = metric_result.get("drift_by_columns", {})
                for col_name, col_info in drift_by_columns.items():
                    if col_info.get("drift_detected", False):
                        drift_result["drifted_features"].append(
                            {
                                "feature": col_name,
                                "drift_score": col_info.get("drift_score", None),
                                "stattest_name": col_info.get("stattest_name", None),
                            }
                        )
                break

        return drift_result

    def generate_report(
        self,
        current_data: pd.DataFrame,
        output_path: Path | None = None,
        is_preprocessed: bool = False,
    ) -> str:
        """
        Generate a detailed drift report as HTML.

        Args:
            current_data: The current production data to compare
            output_path: Where to save the HTML report
            is_preprocessed: If True, skip preprocessing (data is already transformed).
        """
        if not self._loaded:
            self.load_reference()

        # Preprocess current data if needed
        if not is_preprocessed:
            curr = self.preprocessor.transform(current_data)
        else:
            curr = current_data

        ref = self.reference_data

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=curr)

        if output_path is None:
            output_path = Path("drift_report.html")

        report.save_html(str(output_path))
        logger.info(f"Drift report saved to {output_path}")
        return str(output_path)


drift_monitor = DriftMonitor()
