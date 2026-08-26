import csv
import json
import os
import re
import shutil
import struct

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# fixed config ：with after need when in
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAT_FILE = os.path.join(
    SCRIPT_DIR,
    "zhuxin.dat",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "DAT解析结果")

# preview No. 30 s one frame 。
PREVIEW_SECOND = 30.0

# perfusion image display range ：only affects image slice and curve ，not save NPY count value 。
PERFUSION_DISPLAY_MIN = 0
PERFUSION_DISPLAY_MAX = 1600.0

# full-length export save all 2641 frame four NPY data 。
EXPORT_WHOLE_RECORDING = True

# only export one when between slice segment when make use down two when between 。
SEGMENT_START_SECOND = 30.0
SEGMENT_END_SECOND = 30.1

# ROI use Python then ：[ point , point )， point in image left on 。
# when before default take image in 50×50 pixel ， preview image after again 。
ROI_X_START = 249
ROI_X_END = 299
ROI_Y_START = 298
ROI_Y_END = 348

# full-length export and slice segment export with ， use hard 。
EXPORT_SEGMENT = False
EXPORT_ROI_CURVE = True


def decode_fixed_string(raw_bytes):
    """decode file head in length ，and node ten form 。"""
    raw_bytes = raw_bytes.rstrip(b" \x00")
    if not raw_bytes:
        return ""

    for encoding in ("utf-8", "gb18030"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode("ascii", errors="backslashreplace")


def read_header(dat_path):
    """read PIMSoft DAT file head ，and compute each data region node 。"""
    file_size = os.path.getsize(dat_path)
    with open(dat_path, "rb") as file:
        raw_header = file.read(581)

    if len(raw_header) < 46:
        raise ValueError("文件不足 46 字节，不是完整的 PIMSoft DAT 文件。")

    file_type = decode_fixed_string(raw_header[0:10])
    file_version = struct.unpack_from("<i", raw_header, 10)[0]
    signal_gain = struct.unpack_from("<d", raw_header, 14)[0]
    coherence_factor = struct.unpack_from("<d", raw_header, 22)[0]
    total_image_count_float = struct.unpack_from("<d", raw_header, 30)[0]
    image_width, image_height = struct.unpack_from("<ii", raw_header, 38)

    if file_type.strip() != "PSI":
        raise ValueError(f"文件类型为 {file_type!r}，不是 PSI 二进制文件。")
    if file_version not in (1, 2, 3):
        raise ValueError(f"暂不支持文件版本 {file_version}。")
    if not total_image_count_float.is_integer():
        raise ValueError("文件头中的图像数量不是整数。")

    total_image_count = int(total_image_count_float)
    if total_image_count % 2 != 0:
        raise ValueError("图像总数必须为偶数，因为每帧包含方差和光强两幅图。")

    frame_count = total_image_count // 2
    pixel_count = image_width * image_height
    file_header_size = {1: 46, 2: 540, 3: 581}[file_version]
    frame_header_size = frame_count * 32 if file_version == 3 else 0
    variance_offset = file_header_size + frame_header_size
    intensity_offset = variance_offset + frame_count * pixel_count * 8
    expected_file_size = intensity_offset + frame_count * pixel_count * 8

    header = {
        "file_type": file_type,
        "file_version": file_version,
        "signal_gain": signal_gain,
        "coherence_factor": coherence_factor,
        "total_image_count": total_image_count,
        "frame_count": frame_count,
        "image_width": image_width,
        "image_height": image_height,
        "pixel_count_per_frame": pixel_count,
        "file_header_size_bytes": file_header_size,
        "frame_headers_size_bytes": frame_header_size,
        "variance_offset_bytes": variance_offset,
        "intensity_offset_bytes": intensity_offset,
        "actual_file_size_bytes": file_size,
        "expected_file_size_bytes": expected_file_size,
        "file_size_matches_specification": file_size == expected_file_size,
    }

    if file_version >= 2:
        header.update(
            {
                "recording_name": decode_fixed_string(raw_header[46:126]),
                "instrument_serial": decode_fixed_string(raw_header[126:146]),
                "recording_time": decode_fixed_string(raw_header[146:166]),
                "measurement_duration_seconds": struct.unpack_from(
                    "<d", raw_header, 166
                )[0],
                "project_name": decode_fixed_string(raw_header[174:254]),
                "site_name": decode_fixed_string(raw_header[254:334]),
                "subject_name": decode_fixed_string(raw_header[334:414]),
                "operator_name": decode_fixed_string(raw_header[414:494]),
                "measurement_distance_mm": struct.unpack_from(
                    "<d", raw_header, 494
                )[0],
                "frame_rate_text": decode_fixed_string(raw_header[502:522]),
                "resolution_mm_per_pixel": struct.unpack_from(
                    "<d", raw_header, 522
                )[0],
                "point_density": decode_fixed_string(raw_header[530:540]),
            }
        )

    if file_version >= 3:
        header.update(
            {
                "instrument_type": decode_fixed_string(raw_header[540:560]),
                "zoom_enabled": bool(raw_header[560]),
                "zoom_setup_percent": struct.unpack_from("<d", raw_header, 561)[0],
                "focus_setup": struct.unpack_from("<d", raw_header, 569)[0],
                "averaging": struct.unpack_from("<i", raw_header, 577)[0],
            }
        )

    header["frame_rate_hz"] = determine_frame_rate(header)

    if not header["file_size_matches_specification"]:
        # file can by truncate （ in or not ），by data amount new estimate frame count 。
        # variance region and region each frame_count * pixel_count * 8 node ，two same when 。
        data_bytes = file_size - header["variance_offset_bytes"]
        actual_frame_count = data_bytes // (pixel_count * 8 * 2)
        actual_total_images = actual_frame_count * 2
        if actual_frame_count == 0:
            raise ValueError(
                "文件中没有足够的图像数据，可能严重损坏。"
            )
        print(
            f"[WARNING] 文件大小与规范不一致（实际 {file_size}，预期 {expected_file_size}）。\n"
            f"  可能是录制中断或文件拷贝不完整。\n"
            f"  头部声称 {frame_count} 帧，按实际数据量修正为 {actual_frame_count} 帧。"
        )
        header["frame_count"] = actual_frame_count
        header["total_image_count"] = actual_total_images
        header["frame_headers_size_bytes"] = (
            actual_frame_count * 32 if file_version == 3 else 0
        )
        header["intensity_offset_bytes"] = (
            header["variance_offset_bytes"]
            + actual_frame_count * pixel_count * 8
        )
        header["expected_file_size_bytes"] = (
            header["intensity_offset_bytes"]
            + actual_frame_count * pixel_count * 8
        )
        header["file_size_matches_specification"] = True

    return header


def determine_frame_rate(header):
    """ from frame in take count ，failure when again use frame count and when length estimate 。"""
    frame_rate_text = header.get("frame_rate_text", "")
    match = re.search(r"\d+(?:\.\d+)?", frame_rate_text)
    if match:
        return float(match.group())

    duration = header.get("measurement_duration_seconds", 0.0)
    frame_count = header["frame_count"]
    if duration > 0 and frame_count > 1:
        return (frame_count - 1) / duration

    raise ValueError("无法从文件头确定帧率。")


def open_raw_data_maps(dat_path, header):
    """use memory variance region and region ， one time read large file 。"""
    raw_value_count = header["frame_count"] * header["pixel_count_per_frame"]

    variance_map = np.memmap(
        dat_path,
        dtype="<f8",
        mode="r",
        offset=header["variance_offset_bytes"],
        shape=(raw_value_count,),
    )
    intensity_map = np.memmap(
        dat_path,
        dtype="<f8",
        mode="r",
        offset=header["intensity_offset_bytes"],
        shape=(raw_value_count,),
    )
    return variance_map, intensity_map


def get_frame(raw_map, frame_index, header):
    """by frame no. read image ，and set file still as height ×width matrix 。"""
    if frame_index < 0 or frame_index >= header["frame_count"]:
        raise IndexError(
            f"帧号 {frame_index} 超出范围 0~{header['frame_count'] - 1}。"
        )

    pixel_count = header["pixel_count_per_frame"]
    start = frame_index * pixel_count
    end = start + pixel_count
    return raw_map[start:end].reshape(
        (header["image_height"], header["image_width"]),
        order="F",
    )


def read_frame_parameters(dat_path, header):
    """read v3 file in each frame distance off 、scale 、 point and empty between part 。"""
    if header["file_version"] < 3:
        return None

    return np.memmap(
        dat_path,
        dtype="<f8",
        mode="r",
        offset=header["file_header_size_bytes"],
        shape=(header["frame_count"], 4),
    )


def calculate_contrast(variance, intensity, coherence_factor):
    """ 2， variance and compute speckle contrast 。"""
    variance = np.asarray(variance)
    intensity = np.asarray(intensity)

    with np.errstate(divide="ignore", invalid="ignore"):
        contrast = (
            np.sign(variance)
            * coherence_factor
            * np.sqrt(np.abs(variance))
            / intensity
        )
    return contrast


def calculate_perfusion(variance, intensity, coherence_factor, signal_gain):
    """ 4 compute perfusion amount ，and by 5 set on as 3000。"""
    contrast = calculate_contrast(variance, intensity, coherence_factor)

    with np.errstate(divide="ignore", invalid="ignore"):
        perfusion = signal_gain * (1.0 / contrast - 1.0)

    return np.minimum(perfusion, 3000.0)


def second_to_frame_range(start_second, end_second, header):
    """set s count region between as left right frame no. region between 。"""
    if start_second < 0 or end_second <= start_second:
        raise ValueError("时间区间必须满足 0 <= 开始时间 < 结束时间。")

    frame_rate = header["frame_rate_hz"]
    start_frame = int(np.floor(start_second * frame_rate))
    end_frame = int(np.ceil(end_second * frame_rate))
    start_frame = max(0, min(start_frame, header["frame_count"]))
    end_frame = max(0, min(end_frame, header["frame_count"]))

    if start_frame >= end_frame:
        raise ValueError("所选时间区间内没有可导出的帧。")

    return start_frame, end_frame


def robust_limits(image, lower_percentile=1.0, upper_percentile=99.0):
    """use value percentile generate display range ， few point for 。"""
    finite_values = np.asarray(image)[np.isfinite(image)]
    if finite_values.size == 0:
        return 0.0, 1.0

    lower = float(np.percentile(finite_values, lower_percentile))
    upper = float(np.percentile(finite_values, upper_percentile))
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def save_preview(
    output_dir,
    frame_index,
    variance_frame,
    intensity_frame,
    contrast_frame,
    perfusion_frame,
    header,
):
    """save same one frame variance 、 、contrast and perfusion amount four preview image 。"""
    os.makedirs(output_dir, exist_ok=True)

    display_items = [
        ("原始方差 Variance", variance_frame, "coolwarm", None, None),
        ("原始光强 Intensity", intensity_frame, "gray", None, None),
        ("散斑对比度 Contrast", contrast_frame, "coolwarm", None, None),
        (
            f"灌注量 Perfusion（色条 {PERFUSION_DISPLAY_MIN:g}~"
            f"{PERFUSION_DISPLAY_MAX:g}）",
            perfusion_frame,
            "turbo",
            PERFUSION_DISPLAY_MIN,
            PERFUSION_DISPLAY_MAX,
        ),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (title, image, color_map, fixed_lower, fixed_upper) in zip(
        axes.ravel(),
        display_items,
    ):
        if fixed_lower is None or fixed_upper is None:
            lower, upper = robust_limits(image)
        else:
            lower, upper = fixed_lower, fixed_upper

        shown = axis.imshow(
            image,
            cmap=color_map,
            vmin=lower,
            vmax=upper,
            origin="upper",
        )
        axis.set_title(title)
        axis.set_xlabel("X（列）")
        axis.set_ylabel("Y（行）")
        axis.add_patch(
            Rectangle(
                (ROI_X_START, ROI_Y_START),
                ROI_X_END - ROI_X_START,
                ROI_Y_END - ROI_Y_START,
                fill=False,
                edgecolor="#00ff00",
                linewidth=1.5,
            )
        )
        colorbar_extend = "both" if fixed_lower is not None else "neither"
        figure.colorbar(
            shown,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            extend=colorbar_extend,
        )

    time_second = frame_index / header["frame_rate_hz"]
    figure.suptitle(
        f"第 {frame_index} 帧，时间约 {time_second:.3f} 秒",
        fontsize=15,
    )
    preview_path = os.path.join(
        output_dir,
        f"第{frame_index:04d}帧_四类数据预览.png",
    )
    figure.savefig(preview_path, dpi=180)
    plt.close(figure)
    return preview_path


def export_segment(
    output_dir,
    start_frame,
    end_frame,
    variance_map,
    intensity_map,
    frame_parameters,
    header,
    folder_name=None,
):
    """per-frame export frame region between four data ，full-length export not memory 。"""
    frame_rate = header["frame_rate_hz"]
    start_second = start_frame / frame_rate
    end_second = end_frame / frame_rate
    if folder_name is None:
        folder_name = f"分割片段_{start_second:.3f}s_{end_second:.3f}s"
    segment_dir = os.path.join(
        output_dir,
        folder_name,
    )
    os.makedirs(segment_dir, exist_ok=True)

    frame_count = end_frame - start_frame
    shape = (
        frame_count,
        header["image_height"],
        header["image_width"],
    )

    variance_output = np.lib.format.open_memmap(
        os.path.join(segment_dir, "方差_variance.npy"),
        mode="w+",
        dtype=np.float64,
        shape=shape,
    )
    intensity_output = np.lib.format.open_memmap(
        os.path.join(segment_dir, "光强_intensity.npy"),
        mode="w+",
        dtype=np.float64,
        shape=shape,
    )
    contrast_output = np.lib.format.open_memmap(
        os.path.join(segment_dir, "散斑对比度_contrast.npy"),
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    perfusion_output = np.lib.format.open_memmap(
        os.path.join(segment_dir, "灌注_perfusion.npy"),
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    for output_index, source_index in enumerate(range(start_frame, end_frame)):
        variance_frame = get_frame(variance_map, source_index, header)
        intensity_frame = get_frame(intensity_map, source_index, header)
        contrast_frame = calculate_contrast(
            variance_frame,
            intensity_frame,
            header["coherence_factor"],
        )
        perfusion_frame = calculate_perfusion(
            variance_frame,
            intensity_frame,
            header["coherence_factor"],
            header["signal_gain"],
        )

        variance_output[output_index] = variance_frame
        intensity_output[output_index] = intensity_frame
        contrast_output[output_index] = contrast_frame
        perfusion_output[output_index] = perfusion_frame

        completed_count = output_index + 1
        if completed_count % 100 == 0 or completed_count == frame_count:
            print(f"NPY 导出进度：{completed_count}/{frame_count} 帧")

    variance_output.flush()
    intensity_output.flush()
    contrast_output.flush()
    perfusion_output.flush()

    parameter_path = os.path.join(segment_dir, "帧号与帧参数.csv")
    with open(parameter_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "导出数组下标",
                "原始帧号",
                "时间_秒",
                "距离",
                "缩放百分比",
                "焦点值",
                "分辨率_mm每像素",
            ]
        )
        for output_index, source_index in enumerate(range(start_frame, end_frame)):
            if frame_parameters is None:
                parameters = ["", "", "", ""]
            else:
                parameters = frame_parameters[source_index].tolist()
            writer.writerow(
                [
                    output_index,
                    source_index,
                    source_index / frame_rate,
                    *parameters,
                ]
            )

    return segment_dir


def estimate_export_size_bytes(frame_count, header):
    """estimate four NPY count groups need hard node count 。"""
    value_count = frame_count * header["pixel_count_per_frame"]
    raw_data_bytes = value_count * 8 * 2
    derived_data_bytes = value_count * 4 * 2
    return raw_data_bytes + derived_data_bytes


def check_export_disk_space(output_dir, frame_count, header):
    """in full-length export before check empty between ，and 1 GB safe residual amount 。"""
    os.makedirs(output_dir, exist_ok=True)
    required_bytes = estimate_export_size_bytes(frame_count, header)
    safety_margin_bytes = 1024**3
    free_bytes = shutil.disk_usage(output_dir).free

    if free_bytes < required_bytes + safety_margin_bytes:
        required_gb = required_bytes / 1_000_000_000
        free_gb = free_bytes / 1_000_000_000
        raise OSError(
            f"整段导出约需 {required_gb:.2f} GB，"
            f"当前磁盘仅剩 {free_gb:.2f} GB。"
        )

    return required_bytes, free_bytes


def validate_roi(header):
    """check fixed ROI is in image border boundary inside 。"""
    valid = (
        0 <= ROI_X_START < ROI_X_END <= header["image_width"]
        and 0 <= ROI_Y_START < ROI_Y_END <= header["image_height"]
    )
    if not valid:
        raise ValueError(
            "ROI 超出图像范围；"
            f"图像宽高为 {header['image_width']}×{header['image_height']}。"
        )


def export_roi_curve(
    output_dir,
    variance_map,
    intensity_map,
    header,
):
    """ mean ROI inside variance and ，again compute full-length ROI contrast and perfusion curve 。"""
    validate_roi(header)
    csv_path = os.path.join(output_dir, "中央ROI_全程曲线.csv")
    times = []
    mean_intensities = []
    perfusions = []

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "帧号",
                "时间_秒",
                "ROI平均方差",
                "ROI平均光强",
                "ROI散斑对比度",
                "ROI灌注量",
            ]
        )

        for frame_index in range(header["frame_count"]):
            variance_roi = get_frame(variance_map, frame_index, header)[
                ROI_Y_START:ROI_Y_END,
                ROI_X_START:ROI_X_END,
            ]
            intensity_roi = get_frame(intensity_map, frame_index, header)[
                ROI_Y_START:ROI_Y_END,
                ROI_X_START:ROI_X_END,
            ]

            mean_variance = float(np.mean(variance_roi))
            mean_intensity = float(np.mean(intensity_roi))
            mean_contrast = float(
                calculate_contrast(
                    mean_variance,
                    mean_intensity,
                    header["coherence_factor"],
                )
            )
            mean_perfusion = float(
                calculate_perfusion(
                    mean_variance,
                    mean_intensity,
                    header["coherence_factor"],
                    header["signal_gain"],
                )
            )
            time_second = frame_index / header["frame_rate_hz"]

            writer.writerow(
                [
                    frame_index,
                    time_second,
                    mean_variance,
                    mean_intensity,
                    mean_contrast,
                    mean_perfusion,
                ]
            )
            times.append(time_second)
            mean_intensities.append(mean_intensity)
            perfusions.append(mean_perfusion)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(times, mean_intensities, linewidth=1.0, color="#333333")
    axes[0].set_ylabel("ROI 平均光强")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, perfusions, linewidth=1.0, color="#d62728")
    axes[1].set_xlabel("时间（秒）")
    axes[1].set_ylabel("ROI 灌注量")
    axes[1].set_ylim(PERFUSION_DISPLAY_MIN, PERFUSION_DISPLAY_MAX)
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        "中央 ROI 全程曲线\n"
        f"X=[{ROI_X_START}, {ROI_X_END})，Y=[{ROI_Y_START}, {ROI_Y_END})"
    )
    curve_path = os.path.join(output_dir, "中央ROI_全程曲线.png")
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)
    return csv_path, curve_path


def make_json_safe(value):
    """set NumPy count value and point count into JSON save form 。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_header_information(output_dir, header, frame_parameters):
    """set file head 、data and frame parameters save as JSON。"""
    os.makedirs(output_dir, exist_ok=True)
    information = dict(header)

    if frame_parameters is not None:
        representative_indices = [
            0,
            header["frame_count"] // 2,
            header["frame_count"] - 1,
        ]
        information["representative_frame_parameters"] = {}
        for frame_index in representative_indices:
            values = frame_parameters[frame_index].tolist()
            information["representative_frame_parameters"][str(frame_index)] = {
                "distance": make_json_safe(values[0]),
                "zoom_percent": make_json_safe(values[1]),
                "focus": make_json_safe(values[2]),
                "resolution_mm_per_pixel": make_json_safe(values[3]),
            }

    json_path = os.path.join(output_dir, "文件头信息.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(information, file, ensure_ascii=False, indent=2)
    return json_path


def main():
    """execution file 、preview 、when between segmentation and ROI curve export 。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    header = read_header(DAT_FILE)
    variance_map, intensity_map = open_raw_data_maps(DAT_FILE, header)
    frame_parameters = read_frame_parameters(DAT_FILE, header)

    header_path = save_header_information(OUTPUT_DIR, header, frame_parameters)

    preview_frame_index = int(round(PREVIEW_SECOND * header["frame_rate_hz"]))
    preview_frame_index = min(preview_frame_index, header["frame_count"] - 1)
    variance_frame = get_frame(variance_map, preview_frame_index, header)
    intensity_frame = get_frame(intensity_map, preview_frame_index, header)
    contrast_frame = calculate_contrast(
        variance_frame,
        intensity_frame,
        header["coherence_factor"],
    )
    perfusion_frame = calculate_perfusion(
        variance_frame,
        intensity_frame,
        header["coherence_factor"],
        header["signal_gain"],
    )
    preview_path = save_preview(
        OUTPUT_DIR,
        preview_frame_index,
        variance_frame,
        intensity_frame,
        contrast_frame,
        perfusion_frame,
        header,
    )

    segment_dir = None
    if EXPORT_SEGMENT:
        start_frame, end_frame = second_to_frame_range(
            SEGMENT_START_SECOND,
            SEGMENT_END_SECOND,
            header,
        )
        segment_dir = export_segment(
            OUTPUT_DIR,
            start_frame,
            end_frame,
            variance_map,
            intensity_map,
            frame_parameters,
            header,
        )

    whole_recording_dir = None
    if EXPORT_WHOLE_RECORDING:
        required_bytes, free_bytes = check_export_disk_space(
            OUTPUT_DIR,
            header["frame_count"],
            header,
        )
        print(
            f"准备导出整段四类 NPY 数据，预计占用 "
            f"{required_bytes / 1_000_000_000:.2f} GB，"
            f"当前可用 {free_bytes / 1_000_000_000:.2f} GB。"
        )
        whole_recording_dir = export_segment(
            OUTPUT_DIR,
            0,
            header["frame_count"],
            variance_map,
            intensity_map,
            frame_parameters,
            header,
            folder_name=f"整段数据_{header['frame_count']}帧",
        )

    roi_outputs = None
    if EXPORT_ROI_CURVE:
        roi_outputs = export_roi_curve(
            OUTPUT_DIR,
            variance_map,
            intensity_map,
            header,
        )

    print("解析完成。")
    print(f"文件版本：v{header['file_version']}")
    print(
        f"尺寸：{header['image_height']}×{header['image_width']}，"
        f"帧数：{header['frame_count']}，"
        f"帧率：{header['frame_rate_hz']:.3f} Hz"
    )
    print(f"文件头：{header_path}")
    print(f"预览图：{preview_path}")
    if segment_dir:
        print(f"分割片段：{segment_dir}")
    if whole_recording_dir:
        print(f"整段 NPY 数据：{whole_recording_dir}")
    if roi_outputs:
        print(f"ROI 数据：{roi_outputs[0]}")
        print(f"ROI 曲线：{roi_outputs[1]}")


if __name__ == "__main__":
    main()
