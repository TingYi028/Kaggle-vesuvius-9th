import os
import json
import shutil
from tqdm import tqdm

# ================= 配置區域 =================
INPUT_BASE_DIR = "./nnUNet_raw"
# 建議修改 ID 以避免覆蓋之前的 Dataset，這裡設為 504 (或是原本的 503)
DATASET_ID = "501"
DATASET_NAME = "Vesuvius3D_Official"
OUTPUT_DIR = f"nnUNet_raw/Dataset{DATASET_ID}_{DATASET_NAME}"

# 預設間距 [Z, Y, X]
DEFAULT_SPACING = [1.0, 1.0, 1.0]

# 來源目錄
train_img_src = os.path.join(INPUT_BASE_DIR, "train_images")
train_lab_src = os.path.join(INPUT_BASE_DIR, "train_labels")
test_img_src = os.path.join(INPUT_BASE_DIR, "test_images")

# nnU-Net 目錄
imagesTr = os.path.join(OUTPUT_DIR, "imagesTr")
labelsTr = os.path.join(OUTPUT_DIR, "labelsTr")
imagesTs = os.path.join(OUTPUT_DIR, "imagesTs")

# 建立目錄
for d in [imagesTr, labelsTr, imagesTs]:
    os.makedirs(d, exist_ok=True)


# ================= 輔助函數 =================

def create_spacing_json(path, spacing):
    """為 3D TIF 建立必要的側邊 JSON 檔案"""
    with open(path.replace(".tif", ".json"), 'w') as f:
        json.dump({"spacing": spacing}, f)


def process_set(src_dir, target_dir, is_label=False):
    """
    簡化後的處理函式：
    - 直接複製檔案 (shutil.copy)
    - 處理檔名 (加上 _0000.tif)
    - 生成 json
    """
    if not os.path.exists(src_dir):
        print(f"目錄不存在跳過: {src_dir}")
        return

    files = [f for f in os.listdir(src_dir) if f.endswith('.tif')]

    # 使用 tqdm 顯示進度
    for filename in tqdm(files, desc=f"複製中 {os.path.basename(src_dir)}"):
        case_id = filename.replace('.tif', '')

        # 設定輸出檔名 (影像需加 _0000 以符合 nnU-Net 規範)
        target_name = f"{case_id}_0000.tif" if not is_label else f"{case_id}.tif"

        src_path = os.path.join(src_dir, filename)
        target_path = os.path.join(target_dir, target_name)

        # --- 核心修改：直接複製檔案，不進行讀取與運算 ---
        try:
            shutil.copy(src_path, target_path)
        except Exception as e:
            print(f"\n[Error] 複製檔案 {filename} 時發生錯誤: {e}")
            continue

        # 生成 3D 間距 JSON
        create_spacing_json(target_path, DEFAULT_SPACING)


# ================= 主流程 =================

if __name__ == "__main__":
    print(f"開始製作 Dataset{DATASET_ID} (Raw Data / 無前處理)...")

    # 1. 處理訓練集影像 (直接複製 + 更名)
    process_set(train_img_src, imagesTr, is_label=False)

    # 2. 處理訓練集標籤 (直接複製)
    process_set(train_lab_src, labelsTr, is_label=True)

    # 3. 處理測試集影像 (直接複製 + 更名)
    process_set(test_img_src, imagesTs, is_label=False)

    # 4. 建立 dataset.json
    # 計算訓練集數量
    num_train = len([f for f in os.listdir(imagesTr) if f.endswith('.tif')])

    dataset_info = {
        "channel_names": {
            "0": "Tiff_Volume_Raw"  # 修改描述，標記為原始資料
        },
        "labels": {
            "background": 0,
            "ink": 1,
            "ignore": 2
        },
        "numTraining": num_train,
        "file_ending": ".tif",
        "ignore_label": 2,
        "overwrite_image_reader_writer": "Tiff3DIO",
        "name": f"Dataset{DATASET_ID}_{DATASET_NAME}",
        "description": "Vesuvius 3D dataset (Raw data without preprocessing)."
    }

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"\n轉換完成！")
    print(f"資料集 ID: {DATASET_ID}")
    print(f"儲存路徑: {OUTPUT_DIR}")
    print(f"請使用 nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity 進行下一步。")