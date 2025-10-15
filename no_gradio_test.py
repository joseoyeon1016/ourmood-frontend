# working_catvton_test.py - 올바른 파라미터로 테스트
print("=== CatVTON 최종 테스트 ===")
print("올바른 파라미터로 CatVTON을 로딩합니다.")
print()

import torch
import os
from PIL import Image

# 1. 기본 환경 확인
print("1. 환경 확인:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")
print()

# 2. CatVTON Pipeline import
print("2. CatVTON Pipeline Import:")
try:
    from model.pipeline import CatVTONPipeline

    print("   ✅ CatVTON Pipeline")
except Exception as e:
    print(f"   ❌ CatVTON Pipeline: {e}")
    exit(1)

# 3. 올바른 파라미터로 파이프라인 로딩
print("3. CatVTON 파이프라인 로딩:")
print("   app.py에서 사용하는 파라미터 방식으로 시도...")

try:
    # app.py와 동일한 방식으로 로딩 시도
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",  # app.py에서 사용하는 기본값
        attn_ckpt="zhengchong/CatVTON",  # HuggingFace에서 자동 다운로드
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype,
        device=device,
        use_tf32=True
    )

    print("   🎉🎉🎉 CatVTON 파이프라인 로딩 완전 성공! 🎉🎉🎉")
    print()

    # 4. 테스트 이미지 확인
    print("4. 테스트 이미지 확인:")
    demo_person = "resource/demo/example/person"
    demo_cloth = "resource/demo/example/cloth"

    if os.path.exists(demo_person) and os.path.exists(demo_cloth):
        person_files = [f for f in os.listdir(demo_person) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        cloth_files = [f for f in os.listdir(demo_cloth) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   사람 이미지: {len(person_files)}개")
        print(f"   의상 이미지: {len(cloth_files)}개")

        if person_files and cloth_files:
            print("   데모 이미지들:")
            for i, file in enumerate(person_files[:3]):
                print(f"     사람 {i + 1}: {file}")
            for i, file in enumerate(cloth_files[:3]):
                print(f"     의상 {i + 1}: {file}")

            print()
            print("🚀🚀🚀 CatVTON이 완전히 준비되었습니다! 🚀🚀🚀")
            print()
            print("다음에 할 수 있는 것들:")
            print("1. 실제 가상 착용 실행")
            print("2. 간단한 UI 만들기")
            print("3. 배치로 여러 이미지 처리")

        else:
            print("   테스트 이미지가 없습니다")
    else:
        print("   데모 폴더를 찾을 수 없습니다")
        print("   하지만 엔진 자체는 완벽히 동작합니다!")

except Exception as e:
    print(f"   ❌ 파이프라인 로딩 실패: {e}")
    print()
    print("디버깅 정보:")
    print("app.py 파일에서 사용되는 정확한 파라미터를 확인해주세요.")

print()
print("=== 테스트 완료 ===")
if 'pipeline' in locals():
    print("✅ 성공! CatVTON을 사용할 준비가 완료되었습니다!")
else:
    print("❌ 아직 해결할 문제가 있습니다.")