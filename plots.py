from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pyjobshop.ProblemData import ProblemData
from pyjobshop.Solution import Solution

from pyjobshop.plot.utils import get_colors as _get_colors


def custom_plot_machine_gantt(
    solution: Solution,
    data: ProblemData,
    resources: list[int] | None = None,
    plot_labels: bool = False,
    ax: Axes | None = None,
):
    """
    Plots a Gantt chart of the solution, where each row represents a machine
    and each bar represents a task processed on that machine.

    When plot_labels=True, the job name is displayed in bold inside each bar.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))

    if resources is None:
        resources = list(range(data.num_resources))

    # Tasks belonging to the same job get the same color.
    task2color = defaultdict(lambda: "grey")
    colors = _get_colors()

    for idx, task in enumerate(data.tasks):
        if task.job is not None:
            task2color[idx] = colors[task.job % len(colors)]

    for idx, sol_task in enumerate(solution.tasks):
        kwargs = {
            "color": task2color[idx],
            "linewidth": 1,
            "edgecolor": "black",
            "alpha": 0.75,
        }

        for res_idx in sol_task.resources:
            if res_idx not in resources:
                continue

            y = resources.index(res_idx)

            ax.barh(
                y,
                sol_task.duration,
                left=sol_task.start,
                **kwargs,
            )

            if plot_labels:
                task = data.tasks[idx]

                # Use the job name when available.
                if task.job is not None:
                    label = (data.jobs[task.job].name or f"Job {task.job}")
                else:
                    label = task.name or f"Task {idx}"

                ax.text(
                    sol_task.start + sol_task.duration / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    clip_on=True,
                )

    break_labeled = False

    for res_idx in resources:
        resource = data.resources[res_idx]

        if not hasattr(resource, "breaks"):
            continue

        for start, end in resource.breaks:
            ax.barh(
                resources.index(res_idx),
                end - start,
                left=start,
                color="red",
                alpha=0.1,
                hatch="///",
                edgecolor="darkred",
                linewidth=0.1,
                label="Break" if not break_labeled else None,
            )
            break_labeled = True

    labels = [data.resources[idx].name or f"Machine {idx}" for idx in resources]

    ax.set_yticks(ticks=range(len(labels)), labels=labels)
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_xlabel("Time")
    ax.set_title("Solution")

    if break_labeled:
        ax.legend(loc="best")
