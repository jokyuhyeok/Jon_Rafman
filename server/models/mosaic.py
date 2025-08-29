import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from skimage import segmentation
import cv2

class RegionBasedCollage:
    def __init__(self, input_folder="input", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.create_folders()

    def create_folders(self):
        """input, output 폴더 생성"""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"폴더 생성 완료: {self.input_folder}, {self.output_folder}")

    def load_images(self):
        """input 폴더에서 모든 이미지 로드"""
        image_files = [f for f in os.listdir(self.input_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if len(image_files) < 2:
            raise ValueError("최소 2개 이상의 이미지가 필요합니다")

        images = []
        for file in image_files:
            img_path = os.path.join(self.input_folder, file)
            img = Image.open(img_path).convert('RGB')
            images.append((img, file))

        print(f"{len(images)}개의 이미지를 로드했습니다")
        return images

    def select_target_and_materials(self, images):
        """타겟 이미지와 재료 이미지들 선택"""
        random.shuffle(images)
        target_img, target_file = images[0]
        material_images = images[1:]

        print(f"타겟 이미지: {target_file}")
        print(f"재료 이미지: {[f[1] for f in material_images]}")

        return target_img, material_images

    def create_voronoi_regions(self, width, height, num_points=15):
        """보로노이 다이어그램으로 영역 분할"""
        # 랜덤 시드 포인트 생성 (가장자리까지 포함)
        points = []
        for _ in range(num_points):
            x = random.randint(0, width)
            y = random.randint(0, height)
            points.append([x, y])

        # 경계 포인트 추가하여 가장자리 처리 (모서리 제외)
        boundary_points = [
            [width//2, 0],           # 상단 중점
            [width, height//2],      # 우측 중점
            [width//2, height],      # 하단 중점
            [0, height//2]           # 좌측 중점
        ]
        points.extend(boundary_points)

        points = np.array(points)

        # 보로노이 다이어그램 생성
        vor = Voronoi(points)

        return vor, len(points)  # 모든 포인트 반환 (경계 포인트 포함)

    def create_region_masks(self, vor, width, height, num_points):
        """보로노이 영역들을 마스크로 변환 (무한 리전 클리핑 포함)"""
        masks = []

        # 각 시드 포인트별로 안정적인 리전 매핑
        for i in range(num_points):
            try:
                # scipy가 계산한 직접 매핑 사용 (포인트 → 리전 인덱스)
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]

                # 빈 리전만 건너뛰기 (무한 리전은 처리함)
                if len(region) == 0:
                    continue

                # 새 마스크 생성 (처음에는 검은색)
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)

                # 리전 꼭짓점 좌표 수집
                vertices = []
                has_infinite = False  # 무한점(-1) 포함 여부

                for vertex_idx in region:
                    if vertex_idx == -1:  # 무한점 발견
                        has_infinite = True
                        continue  # 무한점은 좌표 수집에서 제외

                    # 유한한 꼭짓점의 좌표 가져오기
                    vertex = vor.vertices[vertex_idx]
                    # 캔버스 범위 내로 좌표 제한
                    x = max(0, min(width, vertex[0]))
                    y = max(0, min(height, vertex[1]))
                    vertices.append((x, y))

                # 무한 리전인 경우 캔버스 경계로 확장해서 채우기
                if has_infinite and len(vertices) >= 2:
                    extended_vertices = vertices.copy()

                    # 기존 꼭짓점들이 캔버스 경계에 있는지 확인
                    for x, y in vertices:
                        # 경계에 닿은 점들 주변으로 캔버스 모서리 추가
                        if x <= 5:  # 왼쪽 경계 근처
                            extended_vertices.extend([(0, 0), (0, height)])
                        elif x >= width - 5:  # 오른쪽 경계 근처
                            extended_vertices.extend([(width, 0), (width, height)])

                        if y <= 5:  # 상단 경계 근처
                            extended_vertices.extend([(0, 0), (width, 0)])
                        elif y >= height - 5:  # 하단 경계 근처
                            extended_vertices.extend([(0, height), (width, height)])

                    # 중복 좌표 제거
                    vertices = list(set(extended_vertices))

                # 유효한 다각형인지 확인 (최소 3개 점 필요)
                if len(vertices) >= 3:
                    # 흰색으로 다각형 그리기 (마스크에서 255 = 불투명)
                    draw.polygon(vertices, fill=255)
                    # 찢어진 종이 효과 적용
                    torn_mask = self.add_torn_edges_no_blur(mask)
                    masks.append(torn_mask)

            except (IndexError, KeyError) as e:
                # 개별 포인트 처리 실패시 건너뛰기
                print(f"포인트 {i} 처리 실패: {e}")
                continue

        print(f"유효한 마스크 {len(masks)}개 생성됨 (무한 리전 포함)")
        return masks

    def add_torn_edges_no_blur(self, mask):
        """블러 없는 찢어진 종이 효과 생성"""
        mask_array = np.array(mask)

        # 1. 경계 찾기
        edges = cv2.Canny(mask_array, 50, 150)

        # 2. 경계 영역 확장 (작업할 영역 넓히기)
        edge_zone = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=1)

        # 3. 랜덤 침식 (morphological erosion)
        for _ in range(2):  # 2번만 반복 (블러 없으니 덜 침식)
            random_kernel = np.random.randint(0, 2, (3, 3)).astype(np.uint8)
            mask_array = cv2.erode(mask_array, random_kernel, iterations=1)

        # 4. 경계 부분에 약한 랜덤 노이즈
        noise = np.random.randint(-20, 5, mask_array.shape)  # 더 약한 노이즈
        mask_array = np.where(edge_zone > 0,
                             np.clip(mask_array.astype(int) + noise, 0, 255),
                             mask_array)

        # 5. 블러 제거 - 바로 반환
        result = Image.fromarray(mask_array.astype(np.uint8))

        return result

    def create_simple_regions(self, width, height, num_regions=12):
        """간단한 방식으로 영역 분할 (보로노이가 복잡할 경우 대안)"""
        masks = []

        for _ in range(num_regions):
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)

            # 불규칙한 다각형 생성
            center_x = random.randint(width//4, 3*width//4)
            center_y = random.randint(height//4, 3*height//4)

            points = []
            num_points = random.randint(6, 10)

            for i in range(num_points):
                angle = (2 * np.pi * i) / num_points
                radius = random.randint(50, 150)
                x = center_x + int(radius * np.cos(angle))
                y = center_y + int(radius * np.sin(angle))

                # 캔버스 내부로 제한
                x = max(0, min(width, x))
                y = max(0, min(height, y))
                points.append((x, y))

            draw.polygon(points, fill=255)
            masks.append(self.add_torn_edges_no_blur(mask))

        return masks

    def extract_material_piece(self, material_img, mask):
        """재료 이미지에서 마스크 크기에 맞는 조각 추출"""
        mask_width, mask_height = mask.size
        material_width, material_height = material_img.size

        # 재료 이미지를 마스크 크기로 맞춤
        if material_width < mask_width or material_height < mask_height:
            # 재료 이미지가 더 작으면 리사이즈
            material_resized = material_img.resize((mask_width, mask_height))
        else:
            # 재료 이미지에서 랜덤 영역 추출
            start_x = random.randint(0, material_width - mask_width)
            start_y = random.randint(0, material_height - mask_height)
            material_resized = material_img.crop((
                start_x, start_y,
                start_x + mask_width,
                start_y + mask_height
            ))

        # RGBA로 변환 후 마스크 적용
        material_rgba = material_resized.convert('RGBA')
        material_rgba.putalpha(mask)

        return material_rgba

    def create_cutmix_collage(self, target_img, material_images, output_size=(800, 800)):
        """CutMix 방식의 콜라주 생성"""
        width, height = output_size

        print("영역 분할 중...")
        try:
            # 보로노이 다이어그램으로 영역 분할 시도
            vor, total_points = self.create_voronoi_regions(width, height)
            masks = self.create_region_masks(vor, width, height, total_points)
        except:
            print("보로노이 생성 실패, 간단한 방식 사용")
            masks = self.create_simple_regions(width, height)

        if len(masks) == 0:
            print("마스크 생성 실패, 간단한 방식으로 재시도")
            masks = self.create_simple_regions(width, height)

        print(f"{len(masks)}개 영역 생성됨")

        # **CutMix 핵심: 타겟 이미지를 베이스로 시작**
        result = target_img.resize(output_size).convert('RGBA')
        print("타겟 이미지를 베이스로 설정")

        # 각 영역을 재료 이미지로 교체 (CutMix)
        for i, mask in enumerate(masks):
            try:
                # 재료 이미지 랜덤 선택
                material_img, material_name = random.choice(material_images)
                print(f"영역 {i+1}: {material_name}으로 교체 중...")

                # 재료 이미지에서 해당 영역 크기에 맞는 조각 추출
                material_piece = self.extract_material_piece(material_img, mask)

                # **CutMix 개선: 반투명 블렌딩으로 배경과 재료 합성**
                # PIL의 Image.composite 사용 (더 효율적)

                # 투명도 조절 (70% 블렌딩)
                transparency = 0.7
                adjusted_mask = mask.point(lambda p: int(p * transparency))

                # 재료 이미지와 현재 결과를 마스크 기반으로 합성
                temp_result = Image.composite(material_piece, result, adjusted_mask)
                result = temp_result

                print(f"영역 {i+1} 블렌딩 완료 (투명도: {int(transparency*100)}%)")

            except Exception as e:
                print(f"영역 {i+1} 교체 실패: {e}")
                continue

        return result.convert('RGB')

    def generate_collage(self, output_filename="region_collage.jpg"):
        """영역 기반 콜라주 생성 프로세스"""
        try:
            # 이미지 로드
            images = self.load_images()

            # 타겟과 재료 선택 (이번엔 타겟은 참고용으로만)
            target_img, material_images = self.select_target_and_materials(images)

            # 콜라주 생성 (CutMix 방식)
            print("CutMix 방식 콜라주 생성 중...")
            result = self.create_cutmix_collage(target_img, material_images)

            # 결과 저장
            output_path = os.path.join(self.output_folder, output_filename)
            result.save(output_path, quality=95)

            print(f"콜라주가 생성되었습니다: {output_path}")

            # 결과 시각화
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(target_img)
            plt.title("참고 이미지")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(material_images[0][0])
            plt.title(f"재료 이미지 (예시: {material_images[0][1]})")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(result)
            plt.title("생성된 영역 기반 콜라주")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            return result

        except Exception as e:
            print(f"오류 발생: {e}")
            return None

# 사용 예시
if __name__ == "__main__":
    collage_maker = RegionBasedCollage()

    print("사용법:")
    print("1. input 폴더에 2개 이상의 이미지 파일을 넣어주세요")
    print("2. 아래 코드를 실행하세요:")
    print("   collage_maker.generate_collage()")

    
    result = collage_maker.generate_collage("my_region_collage.jpg")