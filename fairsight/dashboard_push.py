"""
Fairsight Toolkit - Dashboard Integration
========================================

This module handles pushing audit results to SAP HANA Cloud and integrating
with SAP Analytics Cloud for visualization.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import logging
from hdbcli import dbapi
import warnings

logger = logging.getLogger(__name__)

class Dashboard:
    """
    Dashboard integration class for pushing audit results to SAP HANA Cloud
    and creating visualizations in SAP Analytics Cloud.
    """

    def __init__(self, connection_params: Dict[str, str]):
        """
        Initialize Dashboard with SAP HANA connection parameters.

        Args:
            connection_params: Dictionary containing HANA connection details
                              Must include: host, port, user, password, encrypt
        Raises:
            ValueError: If connection_params is not provided or missing required keys
        """
        if not connection_params or not all(k in connection_params for k in ["host", "port", "user", "password"]):
            raise ValueError("connection_params must be provided with keys: host, port, user, password")
        self.connection_params = connection_params
        self.conn = None
        self.default_schema = "FAIRSIGHT"

    def connect(self) -> bool:
        """
        Establish connection to SAP HANA Cloud.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            params = self.connection_params
            self.conn = dbapi.connect(
                address=params["host"],
                port=params["port"],
                user=params["user"],
                password=params["password"],
                encrypt=params.get("encrypt", True)
            )
            logger.info("✅ Successfully connected to SAP HANA Cloud")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to SAP HANA Cloud: {e}")
            self.conn = None
            return False

    def disconnect(self):
        """Close SAP HANA connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("📡 Disconnected from SAP HANA Cloud")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

    def create_tables(self):
        """Create necessary tables in SAP HANA for storing audit results."""
        if not self.conn:
            if not self.connect():
                raise ConnectionError("Cannot connect to SAP HANA")

        try:
            cursor = self.conn.cursor()

            # Create schema (handle already exists error)
            try:
                cursor.execute(f"CREATE SCHEMA {self.default_schema}")
            except dbapi.Error as e:
                if "already exists" in str(e) or "SQL error code: 258" in str(e):
                    pass  # Schema already exists
                else:
                    raise

            # Audit Sessions table
            try:
                cursor.execute(f"""
                    CREATE TABLE {self.default_schema}.AUDIT_SESSIONS (
                        SESSION_ID NVARCHAR(50) PRIMARY KEY,
                        DATASET_NAME NVARCHAR(200),
                        MODEL_NAME NVARCHAR(200), 
                        AUDIT_TYPE NVARCHAR(50),
                        TIMESTAMP TIMESTAMP,
                        STATUS NVARCHAR(20),
                        ETHICAL_SCORE INTEGER,
                        TOTAL_SAMPLES INTEGER,
                        PROTECTED_ATTRIBUTES NCLOB,
                        JUSTIFIED_ATTRIBUTES NCLOB
                    )
                """)
            except dbapi.Error as e:
                if "already exists" in str(e) or "SQL error code: 258" in str(e):
                    pass
                else:
                    raise

            # Bias Detection Results table
            try:
                cursor.execute(f"""
                    CREATE TABLE {self.default_schema}.BIAS_RESULTS (
                        RESULT_ID NVARCHAR(50) PRIMARY KEY,
                        SESSION_ID NVARCHAR(50),
                        ATTRIBUTE_NAME NVARCHAR(100),
                        METRIC_NAME NVARCHAR(100),
                        METRIC_VALUE DOUBLE,
                        THRESHOLD_VALUE DOUBLE,
                        IS_BIASED BOOLEAN,
                        IS_JUSTIFIED BOOLEAN,
                        DETAILS NCLOB,
                        FOREIGN KEY (SESSION_ID) REFERENCES {self.default_schema}.AUDIT_SESSIONS(SESSION_ID)
                    )
                """)
            except dbapi.Error as e:
                if "already exists" in str(e) or "SQL error code: 258" in str(e):
                    pass
                else:
                    raise

            # Fairness Metrics table
            try:
                cursor.execute(f"""
                    CREATE TABLE {self.default_schema}.FAIRNESS_METRICS (
                        METRIC_ID NVARCHAR(50) PRIMARY KEY,
                        SESSION_ID NVARCHAR(50),
                        ATTRIBUTE_NAME NVARCHAR(100),
                        PRECISION_GAP DOUBLE,
                        RECALL_GAP DOUBLE,
                        F1_GAP DOUBLE,
                        DEMOGRAPHIC_PARITY_DIFF DOUBLE,
                        EQUAL_OPPORTUNITY_DIFF DOUBLE,
                        FOREIGN KEY (SESSION_ID) REFERENCES {self.default_schema}.AUDIT_SESSIONS(SESSION_ID)
                    )
                """)
            except dbapi.Error as e:
                if "already exists" in str(e) or "SQL error code: 258" in str(e):
                    pass
                else:
                    raise

            # Model Performance table
            try:
                cursor.execute(f"""
                    CREATE TABLE {self.default_schema}.MODEL_PERFORMANCE (
                        PERFORMANCE_ID NVARCHAR(50) PRIMARY KEY,
                        SESSION_ID NVARCHAR(50),
                        ACCURACY DOUBLE,
                        PRECISION DOUBLE,
                        RECALL DOUBLE,
                        F1_SCORE DOUBLE,
                        ROC_AUC DOUBLE,
                        FOREIGN KEY (SESSION_ID) REFERENCES {self.default_schema}.AUDIT_SESSIONS(SESSION_ID)
                    )
                """)
            except dbapi.Error as e:
                if "already exists" in str(e) or "SQL error code: 258" in str(e):
                    pass
                else:
                    raise

            self.conn.commit()
            logger.info("✅ Successfully created audit tables in SAP HANA")

        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise

    def push(self, audit_results: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Push audit results to SAP HANA Cloud.

        Args:
            audit_results: Complete audit results dictionary
            session_id: Optional session ID, generated if not provided

        Returns:
            str: Session ID of the pushed results
        """
        if not session_id:
            session_id = f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        if not self.conn:
            if not self.connect():
                raise ConnectionError("Cannot connect to SAP HANA")

        try:
            # Ensure tables exist
            self.create_tables()

            cursor = self.conn.cursor()

            # Insert audit session
            self._insert_audit_session(cursor, session_id, audit_results)

            # Insert bias results
            if 'bias' in audit_results:
                self._insert_bias_results(cursor, session_id, audit_results['bias'])

            # Insert fairness metrics
            if 'fairness_metrics' in audit_results:
                self._insert_fairness_metrics(cursor, session_id, audit_results['fairness_metrics'])

            # Insert model performance
            if 'model' in audit_results:
                self._insert_model_performance(cursor, session_id, audit_results['model'])

            self.conn.commit()
            logger.info(f"✅ Successfully pushed audit results to SAP HANA. Session ID: {session_id}")

            return session_id

        except Exception as e:
            logger.error(f"❌ Failed to push results to SAP HANA: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    def _insert_audit_session(self, cursor, session_id: str, results: Dict[str, Any]):
        """Insert audit session information."""

        # Calculate ethical score from results
        ethical_score = self._calculate_ethical_score(results)

        cursor.execute(f"""
            INSERT INTO {self.default_schema}.AUDIT_SESSIONS 
            (SESSION_ID, DATASET_NAME, MODEL_NAME, AUDIT_TYPE, TIMESTAMP, STATUS, 
             ETHICAL_SCORE, TOTAL_SAMPLES, PROTECTED_ATTRIBUTES, JUSTIFIED_ATTRIBUTES)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            session_id,
            results.get('dataset_name', 'Unknown'),
            results.get('model_name', 'Unknown'),
            'FULL' if 'model' in results else 'DATASET_ONLY',
            datetime.now(timezone.utc),
            'COMPLETED',
            ethical_score,
            results.get('total_samples', 0),
            json.dumps(results.get('protected_attributes', [])),
            json.dumps(results.get('justified_attributes', []))
        ])

    def _insert_bias_results(self, cursor, session_id: str, bias_results: List[Dict[str, Any]]):
        """Insert bias detection results."""

        for i, result in enumerate(bias_results):
            result_id = f"{session_id}_bias_{i}"

            cursor.execute(f"""
                INSERT INTO {self.default_schema}.BIAS_RESULTS
                (RESULT_ID, SESSION_ID, ATTRIBUTE_NAME, METRIC_NAME, METRIC_VALUE,
                 THRESHOLD_VALUE, IS_BIASED, IS_JUSTIFIED, DETAILS)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                result_id,
                session_id,
                result.get('attribute', 'Unknown'),
                result.get('metric_name', 'Unknown'),
                result.get('value', 0.0),
                result.get('threshold', 0.0),
                result.get('biased', False),
                result.get('justified', False),
                json.dumps(result.get('details', {}))
            ])

    def _insert_fairness_metrics(self, cursor, session_id: str, fairness_metrics: Dict[str, Any]):
        """Insert fairness metrics."""

        for attr, metrics in fairness_metrics.items():
            metric_id = f"{session_id}_fairness_{attr}"

            cursor.execute(f"""
                INSERT INTO {self.default_schema}.FAIRNESS_METRICS
                (METRIC_ID, SESSION_ID, ATTRIBUTE_NAME, PRECISION_GAP, RECALL_GAP,
                 F1_GAP, DEMOGRAPHIC_PARITY_DIFF, EQUAL_OPPORTUNITY_DIFF)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                metric_id,
                session_id,
                attr,
                metrics.get('precision_gap', 0.0),
                metrics.get('recall_gap', 0.0),
                metrics.get('f1_gap', 0.0),
                metrics.get('demographic_parity_difference', 0.0),
                metrics.get('equal_opportunity_difference', 0.0)
            ])

    def _insert_model_performance(self, cursor, session_id: str, model_results: Dict[str, Any]):
        """Insert model performance metrics."""

        performance_id = f"{session_id}_performance"

        cursor.execute(f"""
            INSERT INTO {self.default_schema}.MODEL_PERFORMANCE
            (PERFORMANCE_ID, SESSION_ID, ACCURACY, PRECISION, RECALL, F1_SCORE, ROC_AUC)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            performance_id,
            session_id,
            model_results.get('accuracy', 0.0),
            model_results.get('precision', 0.0),
            model_results.get('recall', 0.0),
            model_results.get('f1_score', 0.0),
            model_results.get('roc_auc', 0.0)
        ])

    def _calculate_ethical_score(self, results: Dict[str, Any]) -> int:
        """Calculate overall ethical score from audit results."""
        score = 100
        deductions = 0

        # Deduct based on bias findings
        if 'bias' in results:
            for result in results['bias']:
                if result.get('biased', False) and not result.get('justified', False):
                    if result.get('metric_name') == 'Disparate Impact':
                        deductions += 15 if result.get('value', 1) < 0.8 else 7
                    else:
                        deductions += 5

        # Deduct based on fairness metrics
        if 'fairness_metrics' in results:
            for metrics in results['fairness_metrics'].values():
                for gap_name, gap_value in metrics.items():
                    if 'gap' in gap_name.lower() and abs(gap_value) > 0.1:
                        deductions += 4

        return max(score - deductions, 0)

    def get_audit_history(self, limit: int = 10) -> pd.DataFrame:
        """
        Retrieve audit history from SAP HANA.

        Args:
            limit: Number of recent audits to retrieve

        Returns:
            DataFrame with audit history
        """
        if not self.conn:
            if not self.connect():
                raise ConnectionError("Cannot connect to SAP HANA")

        try:
            query = f"""
                SELECT SESSION_ID, DATASET_NAME, MODEL_NAME, AUDIT_TYPE, 
                       TIMESTAMP, ETHICAL_SCORE, TOTAL_SAMPLES
                FROM {self.default_schema}.AUDIT_SESSIONS
                ORDER BY TIMESTAMP DESC
                LIMIT {limit}
            """

            return pd.read_sql(query, self.conn)

        except Exception as e:
            logger.error(f"❌ Failed to retrieve audit history: {e}")
            raise

    def create_sac_dashboard_config(self) -> Dict[str, Any]:
        """
        Generate configuration for SAP Analytics Cloud dashboard.

        Returns:
            Dictionary with SAP Analytics Cloud configuration
        """
        return {
            "dashboard_config": {
                "name": "Fairsight AI Ethics Dashboard",
                "data_sources": [
                    {
                        "name": "HANA_FAIRSIGHT",
                        "type": "HANA_CLOUD", 
                        "connection": {
                            "host": self.connection_params["host"],
                            "schema": self.default_schema
                        },
                        "tables": [
                            "AUDIT_SESSIONS",
                            "BIAS_RESULTS", 
                            "FAIRNESS_METRICS",
                            "MODEL_PERFORMANCE"
                        ]
                    }
                ],
                "widgets": [
                    {
                        "type": "KPI",
                        "title": "Average Ethical Score",
                        "query": "SELECT AVG(ETHICAL_SCORE) FROM AUDIT_SESSIONS"
                    },
                    {
                        "type": "CHART",
                        "title": "Ethical Scores Over Time",
                        "chart_type": "LINE",
                        "x_axis": "TIMESTAMP",
                        "y_axis": "ETHICAL_SCORE",
                        "query": "SELECT TIMESTAMP, ETHICAL_SCORE FROM AUDIT_SESSIONS ORDER BY TIMESTAMP"
                    },
                    {
                        "type": "CHART", 
                        "title": "Bias Detection by Attribute",
                        "chart_type": "BAR",
                        "x_axis": "ATTRIBUTE_NAME",
                        "y_axis": "COUNT",
                        "query": "SELECT ATTRIBUTE_NAME, COUNT(*) as COUNT FROM BIAS_RESULTS WHERE IS_BIASED = TRUE GROUP BY ATTRIBUTE_NAME"
                    }
                ]
            }
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
