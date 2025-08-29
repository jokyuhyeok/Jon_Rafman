import os
import io
import base64
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# 💡 models 폴더에서 RegionBasedCollage 클래스 임포트
from models.mosaic import RegionBasedCollage

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB로 용량 확장

def pil_to_b64(img: Image.Image, fmt="JPEG", quality=90) -> str:
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

@app.get("/test")
def test():
    return {"status": "ok"}

@app.post("/generate-collage")
def generate_collage_api():
    try:
        data = request.get_json(silent=True) or {}
        image_b64_list = data.get('images', [])

        if not image_b64_list:
            return jsonify({"ok": False, "error": "이미지 데이터가 없습니다."}), 400

        image_bytes_list = [base64.b64decode(b64) for b64 in image_b64_list]
        
        # 💡 mosaic.py의 RegionBasedCollage 클래스 인스턴스 생성
        collage_maker = RegionBasedCollage()

        # 이미지 리스트를 콜라주 생성기에 전달
        images = collage_maker.load_images_from_list(image_bytes_list)
        target_img, material_images = collage_maker.select_target_and_materials(images)
        
        # 콜라주 생성
        result_img = collage_maker.create_cutmix_collage(target_img, material_images, output_size=(800, 800))
        
        # 결과 이미지를 Base64로 인코딩
        result_b64 = pil_to_b64(result_img, fmt="PNG")

        # 임시 폴더 정리 (RegionBasedCollage 내부에서 생성한 폴더)
        if os.path.exists("temp_input"): shutil.rmtree("temp_input")
        if os.path.exists("temp_output"): shutil.rmtree("temp_output")

        return jsonify({
            "ok": True,
            "generated_image_b64": result_b64,
            "message": "콜라주 이미지 생성 완료"
        })
    except ValueError as ve:
        return jsonify({"ok": False, "error": str(ve)}), 400
    except Exception as e:
        print(f"API 오류 발생: {e}")
        return jsonify({"ok": False, "error": "내부 서버 오류"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)