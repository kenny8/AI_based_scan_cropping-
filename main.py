from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import time


def deskew_and_crop(image, threshold_value=230, padding=5, tight_crop=True):
    """Выравнивает + максимально плотно обрезает белые края"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Первый порог для нахождения контура и поворота
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    if angle < -45:
        angle += 90

    # Поворот изображения
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # === ВТОРАЯ ОБРАБОТКА НА ПОВЁРНУТОМ ИЗОБРАЖЕНИИ (более плотная) ===
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # Более строгий порог для финальной обрезки
    final_thresh_val = threshold_value + 5 if tight_crop else threshold_value
    _, thresh_rot = cv2.threshold(gray_rot, final_thresh_val, 255, cv2.THRESH_BINARY_INV)

    thresh_rot = cv2.morphologyEx(thresh_rot, cv2.MORPH_CLOSE, kernel)
    thresh_rot = cv2.morphologyEx(thresh_rot, cv2.MORPH_OPEN, kernel)

    contours_rot, _ = cv2.findContours(thresh_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rot:
        return rotated  # fallback

    largest_rot = max(contours_rot, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_rot)

    # Минимальный padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(rotated.shape[1] - x, w + 2 * padding)
    h = min(rotated.shape[0] - y, h + 2 * padding)

    cropped = rotated[y:y + h, x:x + w]
    return cropped


def main():
    root = Path(__file__).parent
    input_dir = root / "HPSCANS"
    output_dir = root / "CROPPED_PHOTOS"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(input_dir.glob("SCAN_*.jpg"))

    if not image_files:
        print("❌ Файлы не найдены")
        return

    total = len(image_files)
    print(f"✅ Найдено {total} фотографий. Плотная обрезка белых краёв...\n")

    processed = 0
    errors = 0
    start_time = time.time()

    # ================== НАСТРОЙКИ ==================
    THRESHOLD = 230  # Основной порог (230-240)
    PADDING = 5  # Отступ в пикселях (уменьши до 3-4, если хочешь ещё плотнее)
    TIGHT_CROP = True  # True = максимально убирает белое
    # ===============================================

    for i, img_path in enumerate(tqdm(image_files, desc="Обработка", unit="фото"), 1):
        try:
            print(f"[{i:3d}/{total}] {img_path.name} → ", end="")

            original = cv2.imread(str(img_path))
            if original is None:
                print("❌ не читается")
                errors += 1
                continue

            cropped = deskew_and_crop(original,
                                      threshold_value=THRESHOLD,
                                      padding=PADDING,
                                      tight_crop=TIGHT_CROP)

            if cropped is None:
                print("⚠️  контур не найден")
                errors += 1
                continue

            save_path = output_dir / img_path.name
            cv2.imwrite(str(save_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

            processed += 1
            orig_h, orig_w = original.shape[:2]
            new_h, new_w = cropped.shape[:2]
            print(f"✅ Готово | {orig_w}x{orig_h} → {new_w}x{new_h}")

        except Exception as e:
            errors += 1
            print(f"❌ Ошибка: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"Всего: {total} | Успешно: {processed} | Ошибок: {errors}")
    print(f"Время: {elapsed:.1f} сек")
    print(f"Результат: {output_dir.resolve()}")
    print("=" * 70)
    print("💡 Если белая рамка всё ещё остаётся — попробуй:")
    print("   • Уменьшить PADDING до 3")
    print("   • Увеличить THRESHOLD до 235-240")
    print("   • Поставить TIGHT_CROP = False (если начнёт резать края)")


if __name__ == "__main__":
    main()