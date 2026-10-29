import os
from PIL import Image, ImageDraw, ImageFont

def generate_sample_manga_page(output_path="input/sample_manga_page.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 800, 1000
    # Background manga kertas putih keabuan
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Gambar 2 Panel Komik
    # Panel 1 (Atas)
    draw.rectangle([40, 40, width - 40, 460], outline=(0, 0, 0), width=5, fill=(248, 248, 248))
    # Gambar sketsa garis dekoratif dalam panel agar menyerupai manga sungguhan
    draw.line([40, 460, width - 40, 40], fill=(220, 220, 220), width=2)
    draw.line([40, 40, width - 40, 460], fill=(220, 220, 220), width=2)

    # Panel 2 (Bawah)
    draw.rectangle([40, 500, width - 40, height - 40], outline=(0, 0, 0), width=5, fill=(248, 248, 248))
    draw.line([40, height - 40, width - 40, 500], fill=(220, 220, 220), width=2)

    font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
    font = ImageFont.truetype(font_path, 28) if os.path.exists(font_path) else ImageFont.load_default()

    # Balon Kata 1 di Panel Atas (Kanan)
    # Teks: お前はもう死んでいる
    b1_box = [480, 80, 720, 300]
    draw.ellipse(b1_box, outline=(0, 0, 0), fill=(255, 255, 255), width=4)
    # Ekor balon kata
    draw.polygon([(520, 280), (500, 340), (550, 290)], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(520, 280), (550, 290)], fill=(255, 255, 255), width=4) # tutupi garis dalam

    text1 = "お前はもう\n死んでいる"
    draw.text((540, 160), text1, fill=(0, 0, 0), font=font)

    # Balon Kata 2 di Panel Bawah (Kiri)
    # Teks: 何！？そんなはずはない！
    b2_box = [80, 560, 360, 780]
    draw.ellipse(b2_box, outline=(0, 0, 0), fill=(255, 255, 255), width=4)
    draw.polygon([(300, 750), (330, 810), (340, 740)], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(300, 750), (340, 740)], fill=(255, 255, 255), width=4)

    text2 = "何！？そんなはずはない！"
    draw.text((120, 660), text2, fill=(0, 0, 0), font=font)

    # Balon Kata 3 di Panel Bawah (Kanan)
    # Teks: 俺は海賊王になる男だ！
    b3_box = [450, 680, 720, 920]
    draw.ellipse(b3_box, outline=(0, 0, 0), fill=(255, 255, 255), width=4)
    text3 = "俺は海賊王に\nなる男だ！"
    draw.text((500, 760), text3, fill=(0, 0, 0), font=font)

    img.save(output_path)
    print(f"Sample halaman manga berhasil dibuat di: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_sample_manga_page()
