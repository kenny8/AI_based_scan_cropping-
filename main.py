from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import time


def deskew_and_crop(image, threshold_value=235, padding=8):
    """Выравнивает + обрезает одно изображение"""
    # 1. В grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Порог (чем выше — тем строже к белому фону)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 3. Морфология — убираем шум
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Находим самый большой контур
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    # 5. Получаем минимальный прямоугольник с углом (для поворота)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]

    # Корректируем угол (чтобы всегда поворачивать в нужную сторону)
    if angle < -45:
        angle += 90

    # 6. Поворачиваем оригинальное изображение
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 7. На повёрнутом изображении снова находим контур (уже ровно)
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh_rot = cv2.threshold(gray_rot, threshold_value, 255, cv2.THRESH_BINARY_INV)
    thresh_rot = cv2.morphologyEx(thresh_rot, cv2.MORPH_CLOSE, kernel)
    thresh_rot = cv2.morphologyEx(thresh_rot, cv2.MORPH_OPEN, kernel)

    contours_rot, _ = cv2.findContours(thresh_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rot:
        return None

    largest_rot = max(contours_rot, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_rot)

    # 8. Добавляем небольшой отступ
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(rotated.shape[1] - x, w + 2 * padding)
    h = min(rotated.shape[0] - y, h + 2 * padding)

    # 9. Финальная обрезка
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
    print(f"✅ Найдено {total} фотографий.")
    print("🔄 Включаем выравнивание + плотную обрезку...\n")

    processed = 0
    errors = 0
    start_time = time.time()

    # Настройка (можно менять):
    THRESHOLD = 235  # 220–245: ниже = больше захватывает, выше = строже
    PADDING = 8  # отступ в пикселях (5–15)

    for i, img_path in enumerate(tqdm(image_files, desc="Обработка", unit="фото"), 1):
        try:
            print(f"[{i:3d}/{total}] {img_path.name} → ", end="")

            original = cv2.imread(str(img_path))
            if original is None:
                print("❌ не читается")
                errors += 1
                continue

            cropped = deskew_and_crop(original, threshold_value=THRESHOLD, padding=PADDING)

            if cropped is None:
                print("⚠️  контур не найден")
                errors += 1
                continue

            save_path = output_dir / img_path.name
            cv2.imwrite(str(save_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

            processed += 1
            orig_h, orig_w = original.shape[:2]
            new_h, new_w = cropped.shape[:2]
            print(f"✅ Готово | {orig_w}x{orig_h} → {new_w}x{new_h} (повёрнуто)")

        except Exception as e:
            errors += 1
            print(f"❌ Ошибка: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 ГОТОВО!")
    print("=" * 70)
    print(f"Всего: {total} | Успешно: {processed} | Ошибок: {errors}")
    print(f"Время: {elapsed:.1f} сек")
    print(f"Папка: {output_dir.resolve()}")
    print("=" * 70)
    print("💡 Если белые полоски остались — уменьши THRESHOLD до 230")
    print("💡 Если режет края фото — увеличь PADDING до 15")


if __name__ == "__main__":
    main()