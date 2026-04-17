from pathlib import Path
from rembg import remove, new_session
from PIL import Image
from tqdm import tqdm
import time


def main():
    root = Path(__file__).parent
    input_dir = root / "HPSCANS"
    output_dir = root / "CROPPED_PHOTOS"

    output_dir.mkdir(exist_ok=True)

    # Загружаем модель один раз
    print("⏳ Загрузка нейронной модели (rembg)...")
    session = new_session()  # можно добавить: model_name="isnet-general-use"

    # Получаем список файлов
    image_files = sorted(input_dir.glob("SCAN_*.jpg"))

    if not image_files:
        print("❌ Не найдено файлов SCAN_*.jpg в папке HPSCANS/")
        return

    total = len(image_files)
    print(f"✅ Найдено {total} фотографий.")
    print(f"📁 Результат будет сохранён в папку: {output_dir}\n")

    processed = 0
    errors = 0
    start_time = time.time()

    # Основной цикл с tqdm
    for i, img_path in enumerate(tqdm(image_files, desc="Обработка", unit="фото"), 1):
        try:
            # Логирование: какая фотография сейчас обрабатывается
            print(f"\n[{i:3d}/{total}] Обработка: {img_path.name}", end=" → ")

            original = Image.open(img_path).convert("RGB")

            # Удаление фона нейронкой
            output = remove(original, session=session, only_mask=False)

            # Получаем прямоугольную область объекта
            bbox = output.getbbox()
            if bbox is None:
                print("⚠️  Пустая маска (ничего не найдено)")
                errors += 1
                continue

            # Обрезаем оригинальное изображение
            cropped = original.crop(bbox)

            # Сохраняем
            save_path = output_dir / img_path.name
            cropped.save(save_path, quality=95, optimize=True, subsampling=0)

            processed += 1
            percent = (i / total) * 100
            print(f"✅ Готово ({percent:.1f}%)")

        except Exception as e:
            errors += 1
            print(f"❌ Ошибка: {e}")

    # Итоговая статистика
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"Всего файлов:          {total}")
    print(f"Успешно обработано:    {processed}")
    print(f"Ошибок:                {errors}")
    print(f"Время обработки:       {elapsed:.1f} секунд ({elapsed / 60:.1f} минут)")
    print(f"Результат сохранён в:  {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()