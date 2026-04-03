# plot_recorder.py
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional


@dataclass
class PlotSeries:
    """Одна серия данных для графика"""
    name:      str
    plot_type: Literal["line", "band", "marker", "hline"]
    color:     str   = "blue"
    alpha:     float = 1.0
    linewidth: float = 1.5
    linestyle: str   = "solid"
    marker:    str   = "circle"
    size:      float = 8.0
    label:     Optional[str] = None
    data:      List[tuple]   = field(default_factory=list)


class PlotRecorder:
    """
    Записывает индикаторы из стратегии для визуализации.
    Всё рисуется на основном графике цены.
    """

    def __init__(self):
        self.series: Dict[str, PlotSeries] = {}

    def line(
        self,
        name:      str,
        value:     float,
        time:      int,
        color:     str   = "#2196F3",
        linewidth: float = 1.5,
        linestyle: str   = "solid",
        alpha:     float = 1.0,
        label:     str   = None,
    ):
        """Линия (MA, EMA, reservation price и т.п.)"""
        if name not in self.series:
            self.series[name] = PlotSeries(
                name=name,
                plot_type="line",
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                label=label or name,
            )
        self.series[name].data.append((time, value))

    def band(
        self,
        name:  str,
        upper: float,
        lower: float,
        time:  int,
        color: str   = "#9C27B0",
        alpha: float = 0.2,
    ):
        """Полоса / канал (Bollinger, spread band и т.п.)"""
        if name not in self.series:
            self.series[name] = PlotSeries(
                name=name,
                plot_type="band",
                color=color,
                alpha=alpha,
                label=name,
            )
        self.series[name].data.append((time, upper, lower))

    def marker(
        self,
        name:   str,
        price:  float,
        time:   int,
        marker: str   = "circle",
        color:  str   = "#4CAF50",
        size:   float = 10.0,
        label:  str   = None,
    ):
        """Маркер / точка (сигналы входа, выхода и т.п.)"""
        if name not in self.series:
            self.series[name] = PlotSeries(
                name=name,
                plot_type="marker",
                color=color,
                marker=marker,
                size=size,
                label=label or name,
            )
        self.series[name].data.append((time, price))

    def hline(
        self,
        name:      str,
        price:     float,
        color:     str   = "#FF5722",
        linestyle: str   = "dashed",
        linewidth: float = 1.0,
        alpha:     float = 0.7,
        label:     str   = None,
    ):
        """Горизонтальная линия (уровни поддержки/сопротивления и т.п.)"""
        # hline всегда перезаписывается последним значением
        self.series[name] = PlotSeries(
            name=name,
            plot_type="hline",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label or name,
            data=[(price,)],
        )

    def has_data(self) -> bool:
        return len(self.series) > 0

    def clear(self, name: str = None):
        if name:
            self.series.pop(name, None)
        else:
            self.series.clear()