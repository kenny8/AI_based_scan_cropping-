from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
from tqdm import tqdm
import time
import cv2


def main():
    root = Path(__file__).parent
    input_dir = root / "CROPPED_PHOTOS"
    output_dir = root / "UPSCALED_2K_PHOTOS"  # ← сюда будут улучшенные фотки

    output_dir.mkdir(exist_ok=True)

    # ================== НАСТРОЙКИ ==================
    MODEL_NAME = "RealESRGAN_x4plus"
    OUTSCALE = 2.0

    # Ссылка на веса модели (стандарт для RealESRGAN)
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    # ===============================================

    print("⏳ Загрузка нейронки Real-ESRGAN...")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_url,  # <--- ТЕПЕРЬ ТУТ ССЫЛКА
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None
    )

    image_files = sorted(input_dir.glob("*.jpg"))

    if not image_files:
        print("❌ В папке CROPPED_PHOTOS ничего не найдено!")
        return

    total = len(image_files)
    print(f"✅ Найдено {total} фотографий. Запускаем увеличение качества до ~2K...\n")

    processed = 0
    start_time = time.time()

    for i, img_path in enumerate(tqdm(image_files, desc="Upscale нейронкой", unit="фото"), 1):
        try:
            print(f"[{i:3d}/{total}] {img_path.name} → ", end="")

            # Загружаем
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print("❌ не читается")
                continue

            # Нейронка делает upscale + улучшение
            output, _ = upsampler.enhance(img, outscale=OUTSCALE)

            # Сохраняем
            save_path = output_dir / img_path.name
            cv2.imwrite(str(save_path), output, [cv2.IMWRITE_JPEG_QUALITY, 98])

            processed += 1
            orig_h, orig_w = img.shape[:2]
            new_h, new_w = output.shape[:2]
            print(f"✅ Готово | {orig_w}x{orig_h} → {new_w}x{new_h} (~2K)")

        except Exception as e:
            print(f"❌ Ошибка: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 УЛУЧШЕНИЕ КАЧЕСТВА ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"Всего обработано: {processed}/{total}")
    print(f"Время: {elapsed:.1f} секунд ({elapsed / 60:.1f} мин)")
    print(f"Результат в папке: {output_dir.resolve()}")
    print("=" * 70)
    print("💡 Хочешь ещё выше качество — поставь OUTSCALE = 4.0")
    print("💡 Хочешь поменьше размер — поставь OUTSCALE = 1.5 или 2.0")


if __name__ == "__main__":
    main()