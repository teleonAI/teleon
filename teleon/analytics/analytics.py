"""
Complete Analytics System.

Combines metrics, dashboards, and reporting.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from collections import defaultdict
from enum import Enum

from teleon.core import StructuredLogger, LogLevel


class ChartType(str, Enum):
    """Chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"


class TimeSeriesData(BaseModel):
    """Time series data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)


class MetricsAggregator:
    """Aggregate and analyze metrics."""
    
    def __init__(self):
        self.data: Dict[str, List[TimeSeriesData]] = defaultdict(list)
        self.logger = StructuredLogger("metrics_aggregator", LogLevel.INFO)
    
    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric data point."""
        data_point = TimeSeriesData(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.data[metric_name].append(data_point)
    
    def query(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TimeSeriesData]:
        """Query metric data."""
        data = self.data.get(metric_name, [])
        
        if start_time:
            data = [d for d in data if d.timestamp >= start_time]
        if end_time:
            data = [d for d in data if d.timestamp <= end_time]
        
        return data
    
    def aggregate(
        self,
        metric_name: str,
        aggregation: str = "sum",
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Aggregate metric data."""
        data = self.data.get(metric_name, [])
        
        if not data:
            return []
        
        # Group by interval
        intervals = defaultdict(list)
        for point in data:
            interval_key = point.timestamp.replace(
                minute=(point.timestamp.minute // interval_minutes) * interval_minutes,
                second=0,
                microsecond=0
            )
            intervals[interval_key].append(point.value)
        
        # Aggregate
        results = []
        for timestamp, values in sorted(intervals.items()):
            if aggregation == "sum":
                agg_value = sum(values)
            elif aggregation == "avg":
                agg_value = sum(values) / len(values)
            elif aggregation == "max":
                agg_value = max(values)
            elif aggregation == "min":
                agg_value = min(values)
            else:
                agg_value = sum(values)
            
            results.append({
                "timestamp": timestamp.isoformat(),
                "value": agg_value
            })
        
        return results


class Widget(BaseModel):
    """Dashboard widget."""
    id: str
    title: str
    chart_type: ChartType
    metric_name: str
    width: int = 12  # Grid width (1-12)


class Dashboard:
    """Interactive dashboard."""
    
    def __init__(self, title: str):
        self.title = title
        self.widgets: List[Widget] = []
        self.logger = StructuredLogger("dashboard", LogLevel.INFO)
    
    def add_widget(self, widget: Widget):
        """Add widget to dashboard."""
        self.widgets.append(widget)
    
    def render(self, aggregator: MetricsAggregator) -> Dict[str, Any]:
        """Render dashboard data."""
        data = {
            "title": self.title,
            "widgets": []
        }
        
        for widget in self.widgets:
            widget_data = aggregator.aggregate(widget.metric_name)
            data["widgets"].append({
                "id": widget.id,
                "title": widget.title,
                "type": widget.chart_type.value,
                "data": widget_data
            })
        
        return data


class Report(BaseModel):
    """Generated report."""
    title: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    sections: List[Dict[str, Any]] = Field(default_factory=list)


class ReportGenerator:
    """Generate analytics reports."""
    
    def __init__(self, aggregator: MetricsAggregator):
        self.aggregator = aggregator
        self.logger = StructuredLogger("report_generator", LogLevel.INFO)
    
    async def generate_report(
        self,
        title: str,
        metrics: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Report:
        """Generate a comprehensive report."""
        report = Report(title=title)
        
        for metric_name in metrics:
            data = self.aggregator.query(metric_name, start_time, end_time)
            
            if data:
                total = sum(d.value for d in data)
                avg = total / len(data)
                
                report.sections.append({
                    "metric": metric_name,
                    "total": total,
                    "average": avg,
                    "count": len(data),
                    "data_points": len(data)
                })
        
        self.logger.info(f"Report generated: {title}")
        return report

