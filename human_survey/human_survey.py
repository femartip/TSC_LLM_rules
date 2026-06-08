import io
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from Utils.load_data import load_dataset

OUTPUT_DIR = Path("generated_surveys")
RULE_RESULTS_DIR = Path("results/llm_results")
CLASS_COLOURS = {0: "#b22222", 1: "#1f4e8c"}
CLASS_NAMES = {0: "Red", 1: "Blue"}
PAGE_SIZE = (8.27 * 72, 11.69 * 72)  # A4 portrait in points.
FONT = "DejaVu Sans"
PARTICIPANTS_PER_GROUP = 2
TEST_INSTANCES_PER_DATASET = 10
FIXED_TEST_INSTANCE_LABELS = {
    "Chinatown": [
        (126, 1),
        (114, 0),
        (44, 0),
        (71, 1),
        (238, 1),
        (196, 1),
        (28, 1),
        (203, 0),
        (141, 0),
        (79, 0),
    ],
    "ECG200": [
        (43, 1),
        (69, 0),
        (22, 0),
        (54, 1),
        (16, 1),
        (32, 1),
        (25, 1),
        (28, 0),
        (15, 0),
        (37, 0),
    ],
}

BACKGROUND_ITEMS = [
    "How familiar are you with machine learning classification?",
    "How familiar are you with reading time-series plots?",
    "How familiar are you with explainable AI or rule-based explanations?",
]

RATING_ITEMS = [
    "I understood the difference between the classes.",
    "The information shown was sufficient.",
    "The explanation was easy to apply.",
    "The explanation matched visible patterns in the plots.",
    "I felt confident in my classifications.",
    "The explanation was too vague.",
    "The explanation contained too much detail.",
]

INTERVIEW_QUESTIONS = [
    "What visual features did you use to classify?",
    "In the task where rules were shown, did the rules help? If yes, which part?",
    "Were any rules confusing, vague, or hard to see in the plots?",
    "Did you rely more on prototypes or rules in the rules condition?",
    "Would you prefer prototypes only, rules only, or both?",
    "Did the rules make you understand the classifier better, or mainly help answer the test?",
]

GROUP_SCHEDULES = {
    "A": [("Chinatown", "prototypes_only"), ("ECG200", "prototypes_plus_rules")],
    "B": [("Chinatown", "prototypes_plus_rules"), ("ECG200", "prototypes_only")],
    "C": [("ECG200", "prototypes_only"), ("Chinatown", "prototypes_plus_rules")],
    "D": [("ECG200", "prototypes_plus_rules"), ("Chinatown", "prototypes_only")],
}

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


@dataclass(frozen=True)
class TestInstance:
    ts_idx: int
    classifier_label: int
    series: np.ndarray


@dataclass(frozen=True)
class DatasetMaterial:
    dataset: str
    prototypes_by_label: dict[int, list[np.ndarray]]
    prototype_ids_by_label: dict[int, list[int]]
    rules_by_label: dict[int, list[str]]
    test_instances: list[TestInstance]
    best_accuracy: float
    repetition: int | None
    y_limits: tuple[float, float]


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    condition: str
    material: DatasetMaterial


@dataclass(frozen=True)
class Participant:
    group: str
    subject_number: int

    @property
    def subject_code(self) -> str:
        return f"{self.group}{self.subject_number:02d}"


def register_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD

    regular_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans-Bold.ttf"),
    ]

    regular_path = next((path for path in regular_candidates if path.exists()), None)
    bold_path = next((path for path in bold_candidates if path.exists()), None)
    if regular_path is None or bold_path is None:
        return

    pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular_path)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold_path)))
    FONT_REGULAR = "DejaVuSans"
    FONT_BOLD = "DejaVuSans-Bold"


register_fonts()


def page_x(x: float) -> float:
    return x * PAGE_SIZE[0]


def page_y(y: float) -> float:
    return y * PAGE_SIZE[1]


def page_w(w: float) -> float:
    return w * PAGE_SIZE[0]


def page_h(h: float) -> float:
    return h * PAGE_SIZE[1]


def pdf_color(value: str) -> colors.Color:
    if value == "black":
        return colors.black
    if value.startswith("#"):
        return colors.HexColor(value)
    try:
        grey = float(value)
        return colors.Color(grey, grey, grey)
    except ValueError:
        return colors.black


def font_name(weight: str) -> str:
    return FONT_BOLD if weight == "bold" else FONT_REGULAR


def add_text(
    pdf: Canvas,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 10,
    weight: str = "normal",
    ha: str = "left",
    va: str = "top",
    wrap: int | None = None,
    color: str = "black",
) -> None:
    if wrap:
        text = "\n".join(textwrap.wrap(text, wrap))

    lines = text.splitlines() or [""]
    x_pt = page_x(x)
    top_y_pt = page_y(y)
    line_height = size * 1.2
    pdf.setFont(font_name(weight), size)
    pdf.setFillColor(pdf_color(color))

    for line_index, line in enumerate(lines):
        if va == "bottom":
            baseline = top_y_pt + line_index * line_height
        else:
            baseline = top_y_pt - size - line_index * line_height

        if ha == "center":
            pdf.drawCentredString(x_pt, baseline, line)
        elif ha == "right":
            pdf.drawRightString(x_pt, baseline, line)
        else:
            pdf.drawString(x_pt, baseline, line)


def add_box(pdf: Canvas, x: float, y: float, w: float, h: float, lw: float = 1.0) -> None:
    pdf.setLineWidth(lw)
    pdf.setStrokeColor(colors.black)
    pdf.rect(page_x(x), page_y(y), page_w(w), page_h(h), stroke=1, fill=0)


def add_line(pdf: Canvas, x1: float, y1: float, x2: float, y2: float, *, color: str = "black", lw: float = 1.0) -> None:
    pdf.setStrokeColor(pdf_color(color))
    pdf.setLineWidth(lw)
    pdf.line(page_x(x1), page_y(y1), page_x(x2), page_y(y2))


def draw_wrapped_lines(
    pdf: Canvas,
    x: float,
    y: float,
    lines: list[str],
    *,
    width: int = 92,
    size: float = 9.2,
    line_height: float = 0.022,
) -> float:
    current_y = y
    for line in lines:
        wrapped = textwrap.wrap(line, width=width) or [""]
        for part in wrapped:
            add_text(pdf, x, current_y, part, size=size)
            current_y -= line_height
        current_y -= 0.006
    return current_y


def checkbox_text(options: list[str]) -> str:
    return "   ".join(f"[ ] {option}" for option in options)


def add_page_chrome(pdf: Canvas, participant: Participant, page_number: int, title: str) -> None:
    add_text(pdf, 0.07, 0.978, title, size=15, weight="bold")
    add_text(
        pdf,
        0.93,
        0.978,
        f"Group {participant.group} | Subject {participant.subject_number}",
        size=9,
        ha="right",
    )
    add_line(pdf, 0.06, 0.952, 0.94, 0.952, color="0.2", lw=0.7)
    add_text(pdf, 0.93, 0.028, f"Page {page_number}", size=8.5, ha="right", va="bottom", color="0.35")


_PLOT_CACHE: dict[tuple[bytes, tuple[float, float], str, int, int], bytes] = {}


def render_series_image(
    series: np.ndarray,
    *,
    width_pt: float,
    height_pt: float,
    y_limits: tuple[float, float],
    color: str,
) -> bytes:
    key = (
        np.asarray(series, dtype=float).tobytes(),
        y_limits,
        color,
        int(round(width_pt)),
        int(round(height_pt)),
    )
    cached = _PLOT_CACHE.get(key)
    if cached is not None:
        return cached

    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72), dpi=160)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_facecolor("white")
    ax.plot(np.arange(len(series)), series, color=color, lw=1.8)
    ax.set_xlim(-0.5, len(series) - 0.5)
    ax.set_ylim(*y_limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.22, lw=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
        spine.set_color("0.2")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    image_bytes = buffer.getvalue()
    _PLOT_CACHE[key] = image_bytes
    return image_bytes


def draw_series_box(
    pdf: Canvas,
    series: np.ndarray,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    y_limits: tuple[float, float],
    color: str,
) -> None:
    width_pt = page_w(w)
    height_pt = page_h(h)
    image_bytes = render_series_image(
        series,
        width_pt=width_pt,
        height_pt=height_pt,
        y_limits=y_limits,
        color=color,
    )
    pdf.drawImage(ImageReader(io.BytesIO(image_bytes)), page_x(x), page_y(y), width=width_pt, height=height_pt)


def split_rule_lines(rule_blob: str) -> list[str]:
    return [line.strip() for line in rule_blob.splitlines() if line.strip()]


def load_best_rule_run(dataset: str) -> dict[str, Any]:
    path = RULE_RESULTS_DIR / f"{dataset}_rulebased_promptV3_3_0_llm_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt-v3 rule results for {dataset}: {path}")

    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not runs:
        raise ValueError(f"No runs found in {path}")

    return max(runs, key=lambda row: (row["accuracy"], -row.get("repetition", 999)))


def compute_y_limits(series_list: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate([np.asarray(series, dtype=float).ravel() for series in series_list])
    min_val = float(np.min(stacked))
    max_val = float(np.max(stacked))
    padding = max((max_val - min_val) * 0.08, 0.03)
    return min_val - padding, max_val + padding


def load_dataset_material(dataset: str) -> DatasetMaterial:
    best_run = load_best_rule_run(dataset)
    train = load_dataset(dataset, data_type="TRAIN_normalized")
    test = load_dataset(dataset, data_type="TEST_normalized")

    prototype_ids_by_label: dict[int, list[int]] = {}
    prototypes_by_label: dict[int, list[np.ndarray]] = {}
    for entry in sorted(best_run["support_examples"], key=lambda item: int(item["class_label"])):
        label = int(entry["class_label"])
        indices = [int(idx) for idx in entry["indices"]]
        prototype_ids_by_label[label] = indices
        prototypes_by_label[label] = [np.asarray(train[idx], dtype=float) for idx in indices]

    rules_by_label = {
        0: split_rule_lines(best_run["extracted_rules"].get("class_0", "")),
        1: split_rule_lines(best_run["extracted_rules"].get("class_1", "")),
    }

    fixed_test_instances = FIXED_TEST_INSTANCE_LABELS.get(dataset)
    if fixed_test_instances is None:
        raise KeyError(f"Missing fixed test instance configuration for {dataset}")
    if len(fixed_test_instances) != TEST_INSTANCES_PER_DATASET:
        raise ValueError(
            f"Expected {TEST_INSTANCES_PER_DATASET} fixed test instances for {dataset}, got {len(fixed_test_instances)}"
        )

    test_instances = [
        TestInstance(
            ts_idx=ts_idx,
            classifier_label=classifier_label,
            series=np.asarray(test[ts_idx], dtype=float),
        )
        for ts_idx, classifier_label in fixed_test_instances
    ]

    y_limits = compute_y_limits(
        [series for series_list in prototypes_by_label.values() for series in series_list]
        + [instance.series for instance in test_instances]
    )

    return DatasetMaterial(
        dataset=dataset,
        prototypes_by_label=prototypes_by_label,
        prototype_ids_by_label=prototype_ids_by_label,
        rules_by_label=rules_by_label,
        test_instances=test_instances,
        best_accuracy=float(best_run["accuracy"]),
        repetition=best_run.get("repetition"),
        y_limits=y_limits,
    )


def build_task_specs() -> dict[str, list[TaskSpec]]:
    materials = {dataset: load_dataset_material(dataset) for dataset in {"Chinatown", "ECG200"}}
    schedules: dict[str, list[TaskSpec]] = {}
    for group, assignments in GROUP_SCHEDULES.items():
        schedules[group] = [
            TaskSpec(dataset=dataset, condition=condition, material=materials[dataset])
            for dataset, condition in assignments
        ]
    return schedules


def task_reference_lines(task: TaskSpec, class_label: int) -> list[str]:
    if task.condition == "prototypes_plus_rules":
        return task.material.rules_by_label[class_label]
    return [f"Use the three example plots above as your reference for the {CLASS_NAMES[class_label]} class."]


def page_intro_and_background(pdf: Canvas, participant: Participant, page_number: int) -> None:
    add_page_chrome(pdf, participant, page_number, "Human Study Survey")

    add_text(pdf, 0.42, 0.925, f"Assigned group: {participant.group}", size=10, weight="bold")
    add_text(pdf, 0.71, 0.925, f"Subject: {participant.subject_number}", size=10, weight="bold")

    add_text(pdf, 0.07, 0.865, "Instructions", size=12, weight="bold")
    add_text(
        pdf,
        0.07,
        0.84,
        "Your task is to predict the reference classifier's class assignment for each time-series plot. There is no feedback during the survey.",
        wrap=105,
    )
    add_text(
        pdf,
        0.07,
        0.785,
        "The classifier outputs are shown as Red and Blue. These colour names are only response labels for this study.",
        wrap=105,
    )
    add_text(
        pdf,
        0.07,
        0.73,
        "For every test plot, choose Red or Blue and then rate how confident you are from 1 (very low) to 5 (very high).",
        wrap=105,
    )

    add_box(pdf, 0.07, 0.08, 0.86, 0.58)
    add_text(pdf, 0.09, 0.635, "Short Background Form", size=12, weight="bold")
    add_text(pdf, 0.09, 0.607, "Please answer on a 1-5 scale.", size=10)

    anchors = [
        "1 Not familiar",
        "2 Slightly familiar",
        "3 Moderately familiar",
        "4 Familiar",
        "5 Very familiar",
    ]
    y = 0.565
    for item in BACKGROUND_ITEMS:
        add_text(pdf, 0.09, y, item, size=10, wrap=60)
        add_text(pdf, 0.09, y - 0.05, checkbox_text(anchors), size=8.4)
        y -= 0.145



def draw_class_block(pdf: Canvas, task: TaskSpec, class_label: int, *, y_top: float) -> None:
    class_name = CLASS_NAMES[class_label]
    add_text(pdf, 0.07, y_top, f"{class_name} reference examples", size=11.5, weight="bold", color=CLASS_COLOURS[class_label])

    x_positions = [0.07, 0.365, 0.66]
    for example_number, (x, series) in enumerate(
        zip(x_positions, task.material.prototypes_by_label[class_label], strict=True),
        start=1,
    ):
        draw_series_box(
            pdf,
            series,
            x=x,
            y=y_top - 0.165,
            w=0.25,
            h=0.13,
            y_limits=task.material.y_limits,
            color=CLASS_COLOURS[class_label],
        )
        add_text(pdf, x + 0.125, y_top - 0.182, f"Example {example_number}", size=8.5, ha="center")

    note_lines = task_reference_lines(task, class_label)
    add_box(pdf, 0.07, y_top - 0.345, 0.86, 0.13)
    note_label = "Reference notes" if task.condition == "prototypes_only" else "Reference notes and rules"
    add_text(pdf, 0.09, y_top - 0.225, note_label, size=9.6, weight="bold")
    draw_wrapped_lines(pdf, 0.09, y_top - 0.248, note_lines, width=104, size=8.6, line_height=0.019)



def page_task_teaching(pdf: Canvas, participant: Participant, task_number: int, task: TaskSpec, page_number: int) -> None:
    add_page_chrome(pdf, participant, page_number, f"Task {task_number}: Reference Material")

    condition_text = (
        "Study the example plots and the short reference notes below. Then classify the 10 test instances on the next pages."
        if task.condition == "prototypes_plus_rules"
        else "Study the example plots below. Then classify the 10 test instances on the next pages."
    )
    add_text(pdf, 0.07, 0.915, condition_text, size=10, wrap=105)

    draw_class_block(pdf, task, 0, y_top=0.815)
    draw_class_block(pdf, task, 1, y_top=0.425)



def draw_test_instance(
    pdf: Canvas,
    instance_number: int,
    instance: TestInstance,
    task: TaskSpec,
    *,
    x_left: float,
    y_top: float,
) -> None:
    add_text(pdf, x_left, y_top, f"{instance_number})", size=11, weight="bold")
    draw_series_box(
        pdf,
        instance.series,
        x=x_left + 0.04,
        y=y_top - 0.12,
        w=0.25,
        h=0.12,
        y_limits=task.material.y_limits,
        color="#404040",
    )
    add_text(pdf, x_left + 0.04, y_top - 0.132, "Label: [ ] Red     [ ] Blue", size=8.4)
    add_text(pdf, x_left + 0.04, y_top - 0.154, "Confidence: [ ] 1   [ ] 2   [ ] 3   [ ] 4   [ ] 5", size=7.2)
    add_line(pdf, x_left, y_top - 0.168, x_left + 0.41, y_top - 0.168, color="0.82", lw=0.6)



def page_task_tests(
    pdf: Canvas,
    participant: Participant,
    task_number: int,
    task: TaskSpec,
    page_number: int,
) -> None:
    add_page_chrome(pdf, participant, page_number, f"Task {task_number}: Test Instances")
    add_text(
        pdf,
        0.07,
        0.915,
        "For each plot, choose Red or Blue and then rate your confidence from 1 (very low) to 5 (very high).",
        size=10,
        wrap=105,
    )

    column_x_positions = [0.07, 0.52]
    y_positions = [0.88, 0.715, 0.55, 0.385, 0.22]
    for column_index, x_left in enumerate(column_x_positions):
        for row_index, y_top in enumerate(y_positions):
            instance_index = column_index * 5 + row_index
            instance = task.material.test_instances[instance_index]
            draw_test_instance(
                pdf,
                instance_index + 1,
                instance,
                task,
                x_left=x_left,
                y_top=y_top,
            )



def add_rating_table(pdf: Canvas, *, y_top: float) -> None:
    x0 = 0.055
    widths = [0.33, 0.13, 0.105, 0.13, 0.105, 0.13]
    row_h = 0.075
    headers = ["Statement", "1 Strongly\ndisagree", "2 Disagree", "3 Neutral", "4 Agree", "5 Strongly\nagree"]

    y = y_top
    x = x0
    for width, header in zip(widths, headers, strict=True):
        add_box(pdf, x, y - row_h, width, row_h)
        add_text(pdf, x + 0.006, y - 0.016, header, size=7.6, weight="bold")
        x += width

    y -= row_h
    for item in RATING_ITEMS:
        x = x0
        for column_index, width in enumerate(widths):
            add_box(pdf, x, y - row_h, width, row_h)
            if column_index == 0:
                add_text(pdf, x + 0.006, y - 0.014, item, size=7.6, wrap=44)
            else:
                add_text(pdf, x + width / 2, y - 0.031, "[ ]", size=9, ha="center")
            x += width
        y -= row_h



def page_task_ratings(pdf: Canvas, participant: Participant, task_number: int, page_number: int) -> None:
    add_page_chrome(pdf, participant, page_number, f"Task {task_number}: Short Ratings")
    add_text(pdf, 0.07, 0.915, "Please answer immediately after finishing this task.", size=10)
    add_rating_table(pdf, y_top=0.86)
    add_text(pdf, 0.07, 0.185, f"Optional note about Task {task_number}:", size=11, weight="bold")
    add_box(pdf, 0.07, 0.055, 0.86, 0.11)



def page_final_interview(pdf: Canvas, participant: Participant, page_number: int) -> None:
    add_page_chrome(pdf, participant, page_number, "Final Interview Guide")
    add_text(pdf, 0.07, 0.915, "Use these prompts after both tasks.", size=10)

    y = 0.86
    for index, question in enumerate(INTERVIEW_QUESTIONS, start=1):
        y = draw_wrapped_lines(pdf, 0.07, y, [f"{index}. {question}"], width=100, size=10.2, line_height=0.024)
        y -= 0.012

    add_text(pdf, 0.07, 0.36, "Interviewer notes:", size=11, weight="bold")
    add_box(pdf, 0.07, 0.06, 0.86, 0.28)



def build_participant_pdf(output_path: Path, participant: Participant, tasks: list[TaskSpec]) -> None:
    pdf = Canvas(str(output_path), pagesize=PAGE_SIZE)
    pdf.setTitle(f"Human Study Survey {participant.subject_code}")
    page_number = 1

    page_intro_and_background(pdf, participant, page_number)
    pdf.showPage()
    page_number += 1

    pdf.showPage()
    page_number += 1

    for task_number, task in enumerate(tasks, start=1):
        if page_number % 2 == 0:
            pdf.showPage()
            page_number += 1

        page_task_teaching(pdf, participant, task_number, task, page_number)
        pdf.showPage()
        page_number += 1

        page_task_tests(pdf, participant, task_number, task, page_number)
        pdf.showPage()
        page_number += 1

        page_task_ratings(pdf, participant, task_number, page_number)
        pdf.showPage()
        page_number += 1

    page_final_interview(pdf, participant, page_number)
    pdf.showPage()
    pdf.save()



def manifest_entry(participant: Participant, tasks: list[TaskSpec]) -> dict[str, Any]:
    return {
        "group": participant.group,
        "subject_number": participant.subject_number,
        "subject_code": participant.subject_code,
        "tasks": [
            {
                "task_number": task_number,
                "dataset": task.dataset,
                "condition": task.condition,
                "best_promptv3_accuracy": task.material.best_accuracy,
                "best_promptv3_repetition": task.material.repetition,
                "prototype_ids_by_label": task.material.prototype_ids_by_label,
                "rules_by_label": task.material.rules_by_label,
                "test_instances": [
                    {
                        "ts_idx": instance.ts_idx,
                        "classifier_label": instance.classifier_label,
                    }
                    for instance in task.material.test_instances
                ],
            }
            for task_number, task in enumerate(tasks, start=1)
        ],
    }



def build_all_surveys() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_specs_by_group = build_task_specs()

    manifest: list[dict[str, Any]] = []
    for group in ["A", "B", "C", "D"]:
        for subject_number in range(1, PARTICIPANTS_PER_GROUP + 1):
            participant = Participant(group=group, subject_number=subject_number)
            output_path = OUTPUT_DIR / f"survey_group_{group}_subject_{subject_number:02d}.pdf"
            build_participant_pdf(output_path, participant, task_specs_by_group[group])
            manifest.append(manifest_entry(participant, task_specs_by_group[group]))
    return manifest



def main() -> None:
    manifest = build_all_surveys()
    manifest_path = OUTPUT_DIR / "survey_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} participant surveys to {OUTPUT_DIR}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
