from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import time


def main():
    root = Path(__file__).parent
    input_dir = root / "HPSCANS"
    output_dir = root / "CROPPED_PHOTOS"

    output_dir.mkdir(exist_ok=True)

    image_files = sorted(input_dir.glob("SCAN_*.jpg"))

    if not image_files:
        print("❌ Не найдено файлов SCAN_*.jpg в HPSCANS/")
        return

    total = len(image_files)
    print(f"✅ Найдено {total} фотографий. Начинаем обрезку по контуру...\n")

    processed = 0
    errors = 0
    start_time = time.time()

    for i, img_path in enumerate(tqdm(image_files, desc="Обработка", unit="фото"), 1):
        try:
            print(f"[{i:3d}/{total}] Обработка: {img_path.name} → ", end="")

            # Читаем изображение
            original = cv2.imread(str(img_path))
            if original is None:
                print("❌ Не удалось прочитать")
                errors += 1
                continue

            # Конвертируем в grayscale
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            # Порог (можно подбирать от 200 до 240)
            # Чем выше значение — тем строже к "белому"
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            # Морфология — убираем мелкий шум
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Находим контуры
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("⚠️  Контуров не найдено")
                errors += 1
                continue

            # Берём самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Добавляем небольшой отступ (padding), чтобы не обрезать края объекта
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(original.shape[1] - x, w + 2 * padding)
            h = min(original.shape[0] - y, h + 2 * padding)

            # Обрезаем оригинал
            cropped = original[y:y + h, x:x + w]

            # Сохраняем
            save_path = output_dir / img_path.name
            cv2.imwrite(str(save_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

            processed += 1
            percent = (i / total) * 100
            print(f"✅ Готово ({percent:.1f}%)  | Размер был: {original.shape[1]}x{original.shape[0]} → стал: {w}x{h}")

        except Exception as e:
            errors += 1
            print(f"❌ Ошибка: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"Всего файлов:       {total}")
    print(f"Успешно:            {processed}")
    print(f"Ошибок:             {errors}")
    print(f"Время:              {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")
    print(f"Результат в:        {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()