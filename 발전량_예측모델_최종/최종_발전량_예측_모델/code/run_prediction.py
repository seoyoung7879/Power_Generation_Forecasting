import sys
import os
import importlib.util

def import_module_from_path(name, relative_path):
    """
    주어진 상대경로에서 모듈을 동적으로 임포트합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.abspath(os.path.join(base_dir, relative_path))

    if not os.path.isfile(full_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {full_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location(name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """
    사용자 선택에 따라 태양광 또는 풍력 발전량 예측 스크립트를 실행하는 메인 함수.
    """
    try:
        # 상대경로 기준으로 모듈 임포트
        solar_module = import_module_from_path(
            "solar_final_model",
            os.path.join("..", "..", "태양광_발전량_예측_모델", "2. 최종 예측 모델", "코드", "solar_final_model.py")
        )
        wind_module = import_module_from_path(
            "wind_final_model",
            os.path.join("..", "..", "풍력_발전량_예측_모델", "3. 최종 예측 모델", "코드", "wind_final_model.py")
        )

        run_solar_prediction = solar_module.main
        run_wind_prediction = wind_module.main

    except Exception as e:
        print("❌ 오류: 예측 스크립트를 불러오는 데 실패했습니다.")
        print(f"(상세 정보: {e})")
        sys.exit(1)

    while True:
        print("\n======================================")
        print("   신재생에너지 발전량 예측 프로그램   ")
        print("======================================")
        print("1. 태양광 발전량 예측 실행")
        print("2. 풍력 발전량 예측 실행")
        print("3. 프로그램 종료")
        print("--------------------------------------")

        choice = input("→ 실행할 메뉴의 번호를 입력하세요 (1, 2, 3): ")

        if choice == '1':
            print("\n>>> 태양광 발전량 예측을 시작합니다. <<<")
            try:
                run_solar_prediction()
            except Exception as e:
                print(f"\n⚠️ 태양광 예측 실행 중 오류가 발생했습니다: {e}")
            print("\n>>> 태양광 발전량 예측이 완료되었습니다. <<<")

        elif choice == '2':
            print("\n>>> 풍력 발전량 예측을 시작합니다. <<<")
            try:
                run_wind_prediction()
            except Exception as e:
                print(f"\n⚠️ 풍력 예측 실행 중 오류가 발생했습니다: {e}")
            print("\n>>> 풍력 발전량 예측이 완료되었습니다. <<<")

        elif choice == '3':
            print("프로그램을 종료합니다. 이용해주셔서 감사합니다.")
            break

        else:
            print("❌ 잘못된 입력입니다. 1, 2, 또는 3 중에서 선택해주세요.")

if __name__ == "__main__":
    main()
